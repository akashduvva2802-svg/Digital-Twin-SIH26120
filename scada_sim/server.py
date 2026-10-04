"""
Synthetic SCADA server for the Baghewala live well.

Runs the true-well plant and the PLC in accelerated simulated time, publishes all tags over
OPC UA, accepts set-point REQUESTS from the twin (validated by the PLC), logs everything to
the historian, and plays a scripted scenario of field events.

  python -m scada_sim.server --db field.db --days 5 --accel 3600 --scenario standard [--no-twin]
"""
import argparse
import asyncio
import json
import time

import numpy as np
from asyncua import Server, ua

from .config import ENDPOINT, NAMESPACE, LIVE_WELL, SCAN_S, TAGS, DEFAULT_ACCEL
from .historian import Historian
from .plant import TrueWell
from .plc import PLCLogic

SCENARIOS = {
    # (sim day, action, value) -- identical for the baseline and the twin runs
    "standard": [
        (1.0, "productivity_scale", 0.45),     # near-wellbore damage: inflow drops -> fluid pound
        (2.0, "cooling_start", None),          # tubing cools + asphaltene deposits -> rod floating risk
        (3.5, "desync", True),                 # load-cell channel out of sync with position
        (3.75, "desync", False),
    ],
    "quiet": [],
}


def _vtype(path):
    if path.endswith("Card_Position") or path.endswith("Card_Load"):
        return [0.0] * 400, ua.VariantType.Double
    if "Text" in path or "Reason" in path or "Recommendation" in path:
        return "", ua.VariantType.String
    return 0.0, ua.VariantType.Double


async def run(args):
    hist = Historian(args.db)
    plant = TrueWell(LIVE_WELL, start_production_day=args.start_day)
    plc = PLCLogic()
    if args.no_twin:
        plc.mode = 0                                   # LOCAL: current practice, no twin
    plant.spm, plant.stroke, plant.run_frac = plc.cfg["default_spm"], plc.cfg["default_stroke_in"], 1.0

    srv = Server()
    await srv.init()
    srv.set_endpoint(ENDPOINT)
    srv.set_server_name("Baghewala synthetic SCADA")
    idx = await srv.register_namespace(NAMESPACE)
    root = await srv.nodes.objects.add_folder(idx, "Baghewala")
    well = await root.add_folder(idx, LIVE_WELL)
    folders, nodes = {}, {}
    for path, unit, writable, desc in TAGS:
        grp, name = path.split("/")
        if grp not in folders:
            folders[grp] = await well.add_folder(idx, grp)
        val, vt = _vtype(path)
        n = await folders[grp].add_variable(idx, name, ua.Variant(val, vt))
        await n.write_attribute(ua.AttributeIds.Description, ua.DataValue(ua.Variant(ua.LocalizedText(f"{desc} [{unit}]"))))
        if writable:
            await n.set_writable()
        nodes[path] = n
    events = sorted(SCENARIOS[args.scenario], key=lambda e: e[0])
    ev_i = 0
    last_seen = {"Twin/SPM_SP": None, "Twin/RunFrac_SP": None, "Twin/Stroke_SP": None,
                 "Twin/Recommendation": "", "Twin/AlarmText": ""}
    end_t = args.days * 86400.0
    next_log, next_daily, n_events_logged = 0.0, 86400.0, 0
    last_liq = 0.0
    cooling = False
    wall0 = time.time()

    async def put(path, value):
        val, vt = _vtype(path)
        await nodes[path].write_value(ua.Variant(value, vt))

    async with srv:
        await put("PLC/Mode", float(plc.mode))
        print(f"[SCADA] up at {ENDPOINT}  mode={'LOCAL' if plc.mode == 0 else 'REMOTE'}  "
              f"accel={args.accel:.0f}x  days={args.days}", flush=True)
        while plant.t < end_t:
            # bounded catch-up: never jump more than one 15-min scan block, so requests written by the
            # twin are applied within a scan even when the CPU is shared (a real scan never "jumps")
            target = min((time.time() - wall0) * args.accel, plant.t + 900.0)
            new_card = False
            while plant.t < min(target, end_t):
                # scenario events
                while ev_i < len(events) and plant.t >= events[ev_i][0] * 86400:
                    _, act, v = events[ev_i]; ev_i += 1
                    if act == "cooling_start":
                        cooling = True
                    elif act == "desync":
                        plant.desync = bool(v)
                    else:
                        plant.w[act] = v
                    hist.log_event(plant.t, LIVE_WELL, "SCENARIO", act, str(v))
                if cooling:                                   # gradual over one day
                    plant.w["rod_friction_scale"] = min(6.5, plant.w["rod_friction_scale"] + 5.5 * SCAN_S / 86400)
                    plant.w["tubing_temp_fraction"] = max(0.40, plant.w["tubing_temp_fraction"] - 0.20 * SCAN_S / 86400)
                cid = plant.card_id
                plant.step(SCAN_S, plc)
                plc.watchdog(plant.t, plant)
                if plant.card_id != cid:
                    new_card = True
                    c = plant.last_card
                    plc.check_card(plant.t, plant, float(np.max(c["load"])), float(np.min(c["load"])))
                    hist.log_card(plant.t, LIVE_WELL, plant.card_id, plant.spm, plant.stroke, c["pos"], c["load"])
                if plant.t >= next_log:
                    s = plant.state
                    hist.log_tags(plant.t, LIVE_WELL, {
                        "SPM": plant.spm, "RunStatus": 1.0 if plant.running else (-1.0 if plant.trip_left > 0 else 0.0),
                        "MotorPower": plant.motor_kw if plant.running else 0.3, "FluidLevel_true_ft": s["h_f_ft"],
                        "OilIn_true_bpd": s["q_oil_in_bpd"], "T_res_true_K": s["T_res"],
                        "Oil_cum_bbl": plant.acc["oil_bbl"], "kWh_cum": plant.acc["kwh"],
                        "ImpactOver_h": plant.acc["impact_over_h"], "RunFrac_SP": plant.run_frac,
                        "PLC_Mode": plc.mode})
                    next_log += 900.0
                if plant.t >= next_daily:
                    liq = plant.acc["oil_bbl"] + plant.acc["water_bbl"]
                    await put("Wellhead/LiquidRate_Allocated", float((liq - last_liq) * plant.rng.normal(1, 0.04)))
                    last_liq = liq
                    await put("Wellhead/FluidLevel_Acoustic", float(plant.state["h_f_ft"] + plant.rng.normal(0, 25)))
                    await put("Wellhead/FluidLevel_Timestamp", float(plant.t))
                    next_daily += 86400.0
            # ---- publish tags -------------------------------------------------------
            s = plant.state
            T_wh = 32 + 0.35 * (s["T_res"] - 305) - 25 * (0.6 - plant.w["tubing_temp_fraction"])
            await put("PLC/SimTime", float(plant.t))
            await put("SRP/SPM", float(plant.spm if plant.running else 0.0))
            await put("SRP/StrokeLength", float(plant.stroke))
            await put("SRP/RunStatus", float(1 if plant.running else (-1 if plant.trip_left > 0 else 0)))
            await put("SRP/RunTimeFrac_24h", float(plant.acc["run_s"] / max(plant.t, 1)))
            await put("SRP/PumpFillage_POC", float(np.mean(plant.fill_hist)) if plant.fill_hist else 0.0)
            await put("VFD/Frequency", float(plant.spm * 60 / 8 if plant.running else 0.0))
            kw = plant.motor_kw if plant.running else 0.3
            await put("VFD/MotorPower", float(kw))
            await put("VFD/MotorCurrent", float(kw * 1000 / (1.732 * 415 * 0.85)))
            await put("VFD/Energy_24h", float(plant.acc["kwh"]))
            await put("Wellhead/TubingPressure", float(95 + plant.rng.normal(0, 2)))
            await put("Wellhead/CasingPressure", float(plant.w["casing_pressure_psi"] + plant.rng.normal(0, 1)))
            await put("Wellhead/Temperature", float(T_wh + plant.rng.normal(0, 0.5)))
            await put("Steam/Phase", 0.0)
            await put("Steam/DaysSinceInjectionEnd", float(plant.days_since_injection_end()))
            await put("PLC/Mode", float(plc.mode))
            await put("PLC/Alarm", float(plc.alarm))
            await put("PLC/AlarmText", plc.alarm_text)
            await put("PLC/LastRejectReason", plc.last_reject)
            if new_card:
                c = plant.last_card
                await put("SRP/Card_Position", [float(x) for x in c["pos"]])
                await put("SRP/Card_Load", [float(x) for x in c["load"]])
                await put("SRP/PPRL", float(np.max(c["load"])))
                await put("SRP/MPRL", float(np.min(c["load"])))
                await put("SRP/CardSPM", float(plant.spm)); await put("SRP/CardStroke", float(plant.stroke))
                await put("SRP/CardTimestamp", float(plant.last_card_t))
                await put("SRP/CardID", float(plant.card_id))
            # ---- read twin requests (the PLC validates) -----------------------------
            if not args.no_twin:
                hb = await nodes["Twin/Heartbeat"].read_value()
                plc.heartbeat(plant.t, hb)
                req = {}
                for k in ["Twin/SPM_SP", "Twin/RunFrac_SP", "Twin/Stroke_SP"]:
                    v = await nodes[k].read_value()
                    if v and v != last_seen[k]:
                        req[k] = v; last_seen[k] = v
                if req:
                    plc.request(plant.t, plant, spm=req.get("Twin/SPM_SP"), run_frac=req.get("Twin/RunFrac_SP"),
                                stroke=req.get("Twin/Stroke_SP"))
                for k in ["Twin/Recommendation", "Twin/AlarmText"]:
                    v = await nodes[k].read_value()
                    if v and v != last_seen[k]:
                        hist.log_event(plant.t, LIVE_WELL, "TWIN", k.split("/")[1], v); last_seen[k] = v
            for t_, kind, text in plc.events[n_events_logged:]:
                hist.log_event(t_, LIVE_WELL, "PLC", kind, text)
            n_events_logged = len(plc.events)
            hist.commit()
            await asyncio.sleep(0.02)
        summary = {"mode": "baseline (no twin)" if args.no_twin else "twin", "days": args.days,
                   "oil_bbl": round(plant.acc["oil_bbl"], 1), "water_bbl": round(plant.acc["water_bbl"], 1),
                   "kwh": round(plant.acc["kwh"], 1),
                   "kwh_per_bbl_oil": round(plant.acc["kwh"] / max(plant.acc["oil_bbl"], 1e-6), 2),
                   "runtime_frac": round(plant.acc["run_s"] / plant.t, 3),
                   "impact_over_limit_h": round(plant.acc["impact_over_h"], 1),
                   "strokes": round(plant.acc["strokes"]), "cards": plant.card_id, **plc.counters,
                   "wall_s": round(time.time() - wall0, 1)}
        hist.log_event(plant.t, LIVE_WELL, "SCADA", "SUMMARY", json.dumps(summary))
        hist.close()
        print("[SCADA] SUMMARY " + json.dumps(summary), flush=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="field.db")
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--accel", type=float, default=DEFAULT_ACCEL)
    ap.add_argument("--start-day", type=float, default=120.0, help="production day of the live well at start")
    ap.add_argument("--scenario", default="standard", choices=list(SCENARIOS))
    ap.add_argument("--no-twin", action="store_true", help="baseline: PLC in LOCAL, current practice")
    ap.add_argument("--summary", default="scada_summary.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
