"""
Digital-twin connector: the real-time integration service between SCADA and the twin.

Startup (from the historian):
  * completion data   -> rod string, pump, tubing, crank holes
  * fluid properties  -> viscosity-temperature curve (Andrade fit), asphaltene content
  * pressure surveys  -> reservoir pressure
  * production / well tests -> productivity correction for this well
  * rod-failure / unsetting history -> failure statistics reported with the twin's context
Live loop (OPC UA):
  heartbeat -> new card? -> PINN diagnosis (warm start) -> state & parameter estimation
  -> model optimiser + measured-card feedback -> approval policy -> set-point REQUEST
  -> PLC validation -> readback confirmation -> audit trail in the historian

  python connector.py --db field.db [--disconnect-day 4.0 --disconnect-hours 6]
"""
import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from asyncua import Client, ua

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import pinn_digital_twin as P  # noqa: E402
from scada_sim.config import ENDPOINT, NAMESPACE, LIVE_WELL, PLC as PLC_CFG  # noqa: E402
from scada_sim.historian import Historian  # noqa: E402

torch.set_num_threads(1)


class TwinService:
    def __init__(self, db, well=LIVE_WELL, log=print):
        self.log = log
        self.hist = Historian(db)
        self.well = well
        self.audit_rows = []
        self.latency = []
        self.cache = {}
        self.friction_mult = 1.0              # online estimate: rod-friction multiplier (asphaltene / cooling)
        self.J_corr = 1.0                     # online estimate: productivity correction
        self.mismatch_streak = 0
        self.pending = None                   # set-point awaiting readback
        self.d_meas = None                    # measured damping (EMA)
        self.last_fill = 1.0
        self.configure()

    # ------------------------------------------------------------------ startup: from the historian
    def configure(self):
        h = self.hist
        comp = json.loads(h.query("SELECT data FROM completion WHERE well=?", (self.well,))[0][0])
        api, v50, asph, vt = h.query("SELECT api, visc_cP_50C, asphaltene_wt_pct, visc_table FROM fluid_properties "
                                     "WHERE well=?", (self.well,))[0]
        vt = json.loads(vt)
        bhp = h.query("SELECT static_bhp_psi FROM pressure_surveys WHERE well=? ORDER BY date DESC LIMIT 1", (self.well,))[0][0]
        self.comp = comp
        self.rod = dict(diam=comp["rod_diam_in"], L=comp["pump_depth_m"] / 0.3048, E=comp["rod_E_psi"],
                        rho=comp["rod_density_lbm_ft3"], spm=8.0, stroke=86.0, sg=0.96)
        # well model: calibrated Baghewala P50 + this well's measured fluid properties
        P.calibrate_baghewala(P.FIELD_CONFIG)
        wp = dict(P.WELL_REGISTRY["BAGHEWALA_P50"])
        T50, T250 = 50 + 273.15, 250 + 273.15
        B = math.log(vt["50C"] / vt["250C"]) / (1 / T50 - 1 / T250)          # Andrade fit to the lab table
        wp["mu_cold"] = vt["50C"] / 1000 * math.exp(B * (1 / wp["T0"] - 1 / T50))
        wp["mu_hot"] = vt["250C"] / 1000 * math.exp(B * (1 / wp["Ts"] - 1 / T250))
        self.wp = wp
        self.cfg = dict(P.FIELD_CONFIG)
        self.cfg.update(pump_depth_m=comp["pump_depth_m"], rod_diam_in=comp["rod_diam_in"], plunger_in=comp["plunger_in"],
                        tubing_id_in=comp["tubing_id_in"], strokes_in=comp["stroke_holes_in"],
                        spm_at_60Hz=comp["spm_at_60Hz"], p_reservoir_Pa=bhp * 6894.757)
        # trained PINNs
        ck = torch.load(os.path.join(HERE, "digital_twin_models.pt"), map_location="cpu")
        self.res = P.ReservoirPINN(); self.res.load_state_dict(ck["reservoir"]); self.res.eval()
        self.cards0 = pd.read_csv(os.path.join(HERE, "srp_dynamometer_microscale.csv"))
        self.srp = P.SRPPINN(self.cards0, damping_per_s=P.SRP_DAMPING_PER_S); self.srp.load_state_dict(ck["srp"])
        # productivity correction from the last 60 days of well tests vs twin prediction
        tests = h.query("SELECT oil_bpd FROM well_tests WHERE well=? ORDER BY date DESC LIMIT 8", (self.well,))
        # failure history context
        fails = h.query("SELECT mode, COUNT(*), AVG(wellhead_T_degC), AVG(spm_before) FROM rod_failures WHERE well=? "
                        "GROUP BY mode", (self.well,))
        uns = h.query("SELECT COUNT(*) FROM pump_unsetting WHERE well=?", (self.well,))[0][0]
        years = len(h.query("SELECT DISTINCT substr(date,1,4) FROM production_daily WHERE well=?", (self.well,)))
        self.startup = {"completion": {k: comp[k] for k in ["pump_depth_m", "rod_diam_in", "plunger_in", "unit"]},
                        "fluid": {"api": api, "visc_cP_50C": v50, "asphaltene_wt_pct": asph,
                                  "andrade_mu_cold_Pa_s": round(wp["mu_cold"], 2), "andrade_mu_hot_Pa_s": round(wp["mu_hot"], 3)},
                        "reservoir_pressure_psi": round(bhp, 0),
                        "recent_well_tests_oil_bpd": round(float(np.mean([t[0] for t in tests])), 1) if tests else None,
                        "failure_history": {"rod_failures_per_well_year": round(sum(f[1] for f in fails) / max(years, 1), 2),
                                            "by_mode": {f[0]: {"count": f[1], "avg_wellhead_T_C": round(f[2], 1),
                                                               "avg_spm_before": round(f[3], 1)} for f in fails},
                                            "pump_unsettings": uns}}
        self.log("[TWIN] configured from historian: " + json.dumps(self.startup))
        self.audit(0.0, "STARTUP", json.dumps(self.startup))

    def audit(self, t, kind, text):
        self.audit_rows.append((t, kind, text))
        self.hist.log_event(t, self.well, "TWIN-AUDIT", kind, text)
        self.hist.commit()

    # ------------------------------------------------------------------ model predictions for "now"
    def reservoir_now(self, tau_days):
        with torch.no_grad():
            p = P.well_params_tensor(self.wp, 1, "cpu")
            o = self.res(torch.tensor([[float(tau_days)]]), p)
        return float(o["T"]), float(o["J_bpd_per_MPa"]) * self.J_corr

    def tubing_mu(self, T_res, wc=0.0):
        """Tubing liquid viscosity: oil (Andrade fit to the lab table) mixed with water using the SAME
        rule as the twin's SRP optimiser (log-mixing) so diagnosis and optimisation are consistent.
        NOTE: heavy-oil emulsions below inversion are usually MORE viscous than oil; the friction
        multiplier estimated online from the cards absorbs that model error."""
        T_tub = max(self.cfg["T_ambient_K"] + self.cfg["tubing_temp_fraction"] * (T_res - self.cfg["T_ambient_K"]),
                    self.wp["T0"])
        mu_o = float(P.andrade_mu(torch.tensor(T_tub), torch.tensor(self.wp["mu_cold"]), torch.tensor(self.wp["mu_hot"]),
                                  torch.tensor(self.wp["T0"]), torch.tensor(self.wp["Ts"])))
        return mu_o ** (1 - wc) * self.cfg["water_visc_Pa_s"] ** wc

    def update_productivity(self, liquid_bpd, tau, t_sim):
        """Daily state update: allocated liquid rate vs model -> productivity correction (EMA)."""
        q_w = float(P.water_rate(torch.tensor(tau - self.wp["t_soak"]),
                                 self.wp["Qs"] * self.wp["t_inj"] * 86400 * P.ROCK["rho_steam"] / 1000 * 6.28981, self.cfg))
        J_prev = self.J_corr
        self.J_corr = 1.0
        _, J = self.reservoir_now(tau)
        q_o_model = J * (self.cfg["p_reservoir_Pa"] - self.cfg["pw_min_Pa"]) / 1e6
        ratio = max(liquid_bpd - q_w, 0.05 * q_o_model) / max(q_o_model, 1e-6)
        self.J_corr = float(np.clip(0.5 * J_prev + 0.5 * ratio, 0.1, 3.0))
        self.audit(t_sim, "STATE", f"allocated liquid {liquid_bpd:.1f} bpd -> productivity correction {self.J_corr:.2f}")

    def measured_damping(self, card, h_f_ft):
        """Rod-friction damping measured from the surface card itself. At mid-stroke (SHM
        acceleration ~ 0) the upstroke-downstroke load gap = fluid load + 2 x friction force.
        Verified against the wave model: within ~10 % for damping >= 2 1/s (the floating regime)."""
        pos, load = card["pos"], card["load"]
        dpos = np.gradient(pos)
        mid, band = 0.5 * (pos.min() + pos.max()), 0.12 * (pos.max() - pos.min())
        near = np.abs(pos - mid) < band
        up, dn = near & (dpos < 0), near & (dpos > 0)
        if up.sum() < 3 or dn.sum() < 3:
            return None
        W_f = 0.340 * 0.96 * self.cfg["plunger_in"] ** 2 * max(self.rod["L"] - h_f_ft, 0.0)
        F = max(float(np.median(load[up]) - np.median(load[dn])) - W_f, 0.0) / 2
        A = math.pi / 4 * self.rod["diam"] ** 2
        W_rod = self.rod["rho"] * A / 144 * self.rod["L"]
        v = math.pi * (card["stroke"] / 12) * card["spm"] / 60
        return F * 32.2 / (W_rod * v)

    # ------------------------------------------------------------------ one decision on a new card
    def decide(self, card, tags):
        t_sim = tags["PLC/SimTime"]
        t0 = time.time()
        rod = dict(self.rod, spm=card["spm"], stroke=card["stroke"])
        T_res, J = self.reservoir_now(tags["Steam/DaysSinceInjectionEnd"])
        q_w = float(P.water_rate(torch.tensor(tags["Steam/DaysSinceInjectionEnd"] - self.wp["t_soak"]),
                                 self.wp["Qs"] * self.wp["t_inj"] * 86400 * P.ROCK["rho_steam"] / 1000 * 6.28981, self.cfg))
        q_o = J * (self.cfg["p_reservoir_Pa"] - self.cfg["pw_min_Pa"]) / 1e6
        wc = q_w / max(q_w + q_o, 1e-6)
        mu_est = self.tubing_mu(T_res, wc)
        damping_model = P.damping_from_viscosity(mu_est, rod["diam"], self.cfg["tubing_id_in"], rod["rho"])
        d_meas = self.measured_damping(card, tags.get("fluid_level_ft", 0.0))
        if d_meas is not None and self.last_fill >= 0.85:          # needs a full pump card
            self.d_meas = d_meas if self.d_meas is None else 0.5 * self.d_meas + 0.5 * d_meas
        damping = max(damping_model, self.d_meas) if self.d_meas is not None else damping_model
        self.friction_mult = damping / damping_model
        n = len(card["load"])
        dfc = pd.DataFrame({"t_sec_within_stroke": np.linspace(0, 60.0 / card["spm"], n, endpoint=False),
                            "surface_position_ft": card["pos"], "surface_load_lbf": card["load"]})
        cold = "mlp" not in self.cache
        diag, _ = P.diagnose_scada_card(self.srp, self.cards0, dfc, rod, damping_per_s=damping,
                                        plunger_in=self.cfg["plunger_in"], iters=400 if cold else 120,
                                        cache=self.cache)
        t_diag = time.time() - t0
        issues = diag.get("card_issues", [])
        sensor_fault = any(i.startswith("energy balance") for i in issues)
        if diag.get("fillage") is not None and not sensor_fault:
            self.last_fill = diag["fillage"]
        # ---- model-based proposal (reservoir PINN inflow + analytic loads) -------------------
        model_sp = P.optimize_srp(T_res, J, q_w, self.wp, self.cfg)
        # ---- measured-card correction (closed loop, safety first) ---------------------------
        cur = {"spm": card["spm"], "stroke_in": card["stroke"], "run_frac": tags.get("run_frac", 1.0)}
        new, action = P.srp_feedback_step(diag, cur, self.cfg, model_setpoint={"spm": model_sp["spm"]})
        # proactive (predictive) layer: when the card shows no safety problem, move toward the model
        # optimum; while the model is flagged uncertain, only the conservative direction (lower SPM)
        if not sensor_fault and not action.startswith("CUT") and not action.startswith("MIN"):
            target = model_sp["spm"]
            uncertain = action.startswith("HOLD + RECALIBRATE")
            if (target < cur["spm"] - 0.05) or (not uncertain and target > cur["spm"] + 0.05):
                new = dict(new, spm=target, run_frac=model_sp["run_frac"] if not uncertain else new.get("run_frac", 1.0))
                action = (f"PREDICTIVE: move toward model optimum {target:.2f} SPM"
                          f"{' (conservative, model flagged uncertain)' if uncertain else ''}")
        # ---- approval policy: bounded auto-approval, otherwise hold ---------------------------
        step = PLC_CFG["spm_max_step"]
        spm_req = float(np.clip(new["spm"], cur["spm"] - step, cur["spm"] + step))
        spm_req = float(np.clip(spm_req, PLC_CFG["spm_min"], PLC_CFG["spm_max"]))
        rf_req = float(np.clip(new.get("run_frac", 1.0), PLC_CFG["run_frac_min"], 1.0))
        write = not action.startswith("HOLD")
        alarm = ""
        if sensor_fault:
            alarm = "SENSOR ALERT: " + "; ".join(issues)
        elif diag["rod_float_margin"] < 0.2:
            alarm = f"ROD FLOATING RISK: min-load margin {diag['rod_float_margin']:.2f}"
        if self.friction_mult > 3.0:
            alarm = (alarm + " | " if alarm else "") + (f"FRICTION x{self.friction_mult:.1f} vs model: possible asphaltene/wax "
                                                        "deposition or colder tubing -- check wellhead T, schedule hot-oil treatment")
        dec = {"t_sim_day": round(t_sim / 86400, 3), "card_id": card["id"], "diag_s": round(t_diag, 2),
               "warm": not cold, "fillage": diag.get("fillage"), "margin": diag["rod_float_margin"],
               "quality": diag["card_quality"], "fluid_load_ratio": diag.get("fluid_load_ratio"),
               "friction_mult": round(self.friction_mult, 2), "model_spm": round(model_sp["spm"], 2),
               "mu_tubing_est": round(mu_est, 3), "wc_est": round(wc, 2),
               "action": action, "spm_cur": round(cur["spm"], 2), "spm_req": round(spm_req, 2), "rf_req": round(rf_req, 2),
               "write": write, "alarm": alarm, "total_s": round(time.time() - t0, 2)}
        self.latency.append(dec["total_s"])
        return dec

    # ------------------------------------------------------------------ async live loop
    async def run(self, disconnect_day=None, disconnect_hours=0.0, selftest=True, poll_s=0.25):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        async with Client(ENDPOINT, timeout=30) as c:
            idx = await c.get_namespace_index(NAMESPACE)
            base = ["0:Objects", f"{idx}:Baghewala", f"{idx}:{self.well}"]

            async def node(path):
                g, nme = path.split("/")
                return await c.nodes.root.get_child(base + [f"{idx}:{g}", f"{idx}:{nme}"])
            names = ["PLC/SimTime", "PLC/Mode", "PLC/LastRejectReason", "SRP/CardID", "SRP/SPM", "SRP/RunStatus",
                     "Steam/DaysSinceInjectionEnd", "SRP/Card_Position", "SRP/Card_Load", "SRP/CardSPM",
                     "SRP/CardStroke", "Twin/SPM_SP", "Twin/RunFrac_SP", "Twin/Heartbeat", "Twin/Recommendation",
                     "Twin/AlarmText", "Twin/Stroke_SP", "Wellhead/Temperature", "VFD/MotorCurrent",
                     "Wellhead/FluidLevel_Acoustic", "Wellhead/LiquidRate_Allocated", "Wellhead/FluidLevel_Timestamp"]
            N = {k: await node(k) for k in names}
            self.log(f"[TWIN] connected to {ENDPOINT} (namespace {idx}); {len(N)} tags bound")
            self.connected = True
            hb, last_card, last_reject = 0, 0, ""
            last_rf = 1.0
            if selftest:                                  # guard test: an out-of-bounds request must be rejected
                t_sim = await N["PLC/SimTime"].read_value()
                await N["Twin/SPM_SP"].write_value(ua.Variant(20.0, ua.VariantType.Double))
                await asyncio.sleep(1.0)
                rej = await N["PLC/LastRejectReason"].read_value()
                self.audit(t_sim, "SELFTEST", f"requested SPM 20 (out of bounds) -> PLC says: '{rej}'")
                self.selftest = rej
                last_reject = rej
            decisions = []
            state = {"hb": hb, "offline": False, "stop": False}

            async def heartbeat_task():            # independent of computation, like a real service
                while not state["stop"]:
                    try:
                        ts = await N["PLC/SimTime"].read_value()
                    except Exception:
                        return
                    off = disconnect_day is not None and disconnect_day * 86400 <= ts < (disconnect_day + disconnect_hours / 24) * 86400
                    if off and not state["offline"]:
                        self.audit(ts, "OFFLINE", f"twin service stopped (test) for {disconnect_hours} h: no heartbeat")
                    if not off and state["offline"]:
                        self.audit(ts, "ONLINE", "twin service restarted, heartbeat resumed")
                    state["offline"] = off
                    if not state["offline"]:
                        state["hb"] += 1
                        try:
                            await N["Twin/Heartbeat"].write_value(ua.Variant(float(state["hb"]), ua.VariantType.Double))
                        except Exception:
                            return
                    await asyncio.sleep(poll_s)
            hb_task = asyncio.create_task(heartbeat_task())
            while True:
                t_sim = await N["PLC/SimTime"].read_value()
                if t_sim < 0 or (hasattr(self, "_end") and t_sim >= self._end):
                    break
                if state["offline"]:
                    await asyncio.sleep(poll_s); continue
                # readback of the last request
                if self.pending:
                    rej = await N["PLC/LastRejectReason"].read_value()
                    spm_now = await N["SRP/SPM"].read_value()
                    if rej != last_reject:
                        self.audit(t_sim, "REJECTED", f"{self.pending} -> {rej}"); last_reject = rej; self.pending = None
                    elif spm_now > 0 and abs(spm_now - self.pending["spm"]) < 1e-3:
                        self.audit(t_sim, "APPLIED", f"readback SPM {spm_now:.2f} matches request"); self.pending = None
                cid = await N["SRP/CardID"].read_value()
                if cid and cid != last_card:
                    last_card = cid
                    card = {"id": int(cid), "pos": np.array(await N["SRP/Card_Position"].read_value()),
                            "load": np.array(await N["SRP/Card_Load"].read_value()),
                            "spm": await N["SRP/CardSPM"].read_value(), "stroke": await N["SRP/CardStroke"].read_value()}
                    tags = {"PLC/SimTime": t_sim, "Steam/DaysSinceInjectionEnd": await N["Steam/DaysSinceInjectionEnd"].read_value(),
                            "run_frac": last_rf, "fluid_level_ft": await N["Wellhead/FluidLevel_Acoustic"].read_value()}
                    liq = await N["Wellhead/LiquidRate_Allocated"].read_value()
                    lts = await N["Wellhead/FluidLevel_Timestamp"].read_value()
                    if liq and lts != getattr(self, "_last_alloc_ts", None):      # new daily allocation
                        self._last_alloc_ts = lts
                        self.update_productivity(liq, tags["Steam/DaysSinceInjectionEnd"], t_sim)
                    loop = asyncio.get_running_loop()
                    dec = await loop.run_in_executor(pool, self.decide, card, tags)
                    decisions.append(dec); self.decisions = decisions
                    self.log("[TWIN] " + json.dumps(dec))
                    rec = (f"day {dec['t_sim_day']}: {dec['action']} | SPM {dec['spm_cur']} -> {dec['spm_req']}, "
                           f"run {dec['rf_req']}")
                    await N["Twin/Recommendation"].write_value(ua.Variant(rec, ua.VariantType.String))
                    await N["Twin/AlarmText"].write_value(ua.Variant(dec["alarm"], ua.VariantType.String))
                    self.audit(t_sim, "DECISION", json.dumps(dec))
                    if dec["write"]:
                        mode = await N["PLC/Mode"].read_value()
                        if mode == 1:
                            await N["Twin/SPM_SP"].write_value(ua.Variant(dec["spm_req"], ua.VariantType.Double))
                            if abs(dec["rf_req"] - last_rf) > 1e-6:
                                await N["Twin/RunFrac_SP"].write_value(ua.Variant(dec["rf_req"], ua.VariantType.Double))
                                last_rf = dec["rf_req"]
                            self.pending = {"spm": dec["spm_req"], "rf": dec["rf_req"]}
                            self.audit(t_sim, "REQUEST", json.dumps(self.pending))
                        else:
                            self.audit(t_sim, "NOT-SENT", "PLC in LOCAL mode")
                await asyncio.sleep(poll_s)
            state["stop"] = True
            await hb_task
            self.decisions = decisions
        pool.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="field.db")
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--disconnect-day", type=float, default=None)
    ap.add_argument("--disconnect-hours", type=float, default=6.0)
    ap.add_argument("--out", default="twin_run.json")
    a = ap.parse_args()
    svc = TwinService(a.db)
    svc._end = a.days * 86400 - 60
    svc.connected = False
    for _ in range(60):                                  # wait for the SCADA server to come up
        try:
            asyncio.run(svc.run(a.disconnect_day, a.disconnect_hours))
            break
        except Exception as e:                           # server not up yet, or closed at end of run
            if svc.connected:
                print(f"[TWIN] SCADA session ended ({type(e).__name__}); shutting down", flush=True)
                break
            time.sleep(1.0)
    json.dump({"startup": svc.startup, "decisions": getattr(svc, "decisions", []),
               "selftest_reject": getattr(svc, "selftest", ""),
               "latency_s": {"median": float(np.median(svc.latency)) if svc.latency else None,
                             "max": float(np.max(svc.latency)) if svc.latency else None, "n": len(svc.latency)}},
              open(a.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
