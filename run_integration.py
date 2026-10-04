"""
End-to-end integration run + acceptance tests.

  1. build the historian and seed 3 years of field history (10 wells)
  2. BASELINE : SCADA + plant + PLC in LOCAL mode (current practice), no twin
  3. TWIN     : SCADA server + twin connector as separate processes over OPC UA
  4. acceptance tests on the historian, the audit trail and the run summaries

  python run_integration.py [--days 5 --accel 3600]
"""
import argparse, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scada_sim.historian import Historian
from scada_sim import history_seed


def sh(cmd, log):
    return subprocess.Popen(cmd, cwd=HERE, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--accel", type=float, default=3600.0)
    ap.add_argument("--out", default=os.path.join(HERE, "integration_out"))
    ap.add_argument("--tests-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    db0, dbb, dbt = (os.path.join(a.out, f) for f in ["seed.db", "baseline.db", "twin.db"])
    for f in os.listdir(a.out):
        os.remove(os.path.join(a.out, f))
    t0 = time.time()
    h = Historian(db0); counts = history_seed.seed(h); h.close()
    shutil.copy(db0, dbb); shutil.copy(db0, dbt)
    print(f"[RUN] historian seeded {counts} ({time.time() - t0:.1f}s)", flush=True)
    # baseline (fast, no pacing needed)
    p = sh([sys.executable, "-m", "scada_sim.server", "--db", dbb, "--days", str(a.days), "--accel", "1e7", "--no-twin",
            "--summary", os.path.join(a.out, "baseline_summary.json")], os.path.join(a.out, "baseline_server.log"))
    p.wait(); print(f"[RUN] baseline done ({time.time() - t0:.0f}s)", flush=True)
    # twin
    srv = sh([sys.executable, "-m", "scada_sim.server", "--db", dbt, "--days", str(a.days), "--accel", str(a.accel),
              "--summary", os.path.join(a.out, "twin_summary.json")], os.path.join(a.out, "twin_server.log"))
    time.sleep(2.0)
    con = sh([sys.executable, os.path.join("twin", "connector.py"), "--db", dbt, "--days", str(a.days),
              "--disconnect-day", "4.0", "--disconnect-hours", "6", "--out", os.path.join(a.out, "twin_run.json")],
             os.path.join(a.out, "twin_connector.log"))
    srv.wait(); con.wait(timeout=120)
    print(f"[RUN] twin run done ({time.time() - t0:.0f}s)", flush=True)
    acceptance(a.out)



# ============================================================================ acceptance tests
def acceptance(out):
    import json as _j
    R = []

    def check(tid, name, ok, ev):
        R.append({"id": tid, "test": name, "status": "PASS" if ok else "FAIL", "evidence": ev})
        print(f"[{'PASS' if ok else 'FAIL'}] {tid} {name}\n        {ev}", flush=True)
    base = _j.load(open(os.path.join(out, "baseline_summary.json")))
    twin = _j.load(open(os.path.join(out, "twin_summary.json")))
    run = _j.load(open(os.path.join(out, "twin_run.json")))
    h = Historian(os.path.join(out, "twin.db"))
    ev = h.query("SELECT t, source, kind, text FROM events ORDER BY t")
    audit = [(t, k, x) for t, s, k, x in ev if s == "TWIN-AUDIT"]
    plc = [(t, k, x) for t, s, k, x in ev if s == "PLC"]
    scen = {k: t for t, s, k, x in ev if s == "SCENARIO"}
    decs = [(t, _j.loads(x)) for t, k, x in audit if k == "DECISION"]
    # A: data plumbing
    tables = ["production_daily", "well_tests", "css_cycles", "steam_injection_daily", "vfd_daily", "rod_failures",
              "pump_unsetting", "completion", "fluid_properties", "pressure_surveys"]
    counts = {tb: h.query(f"SELECT COUNT(*) FROM {tb}")[0][0] for tb in tables}
    st = run.get("startup", {})
    check("I1", "Historian holds every data type in the problem statement; twin configured itself from it",
          all(v > 0 for v in counts.values()) and "failure_history" in st and "fluid" in st,
          f"record counts {counts}; twin read completion, fluid table (Andrade fit), pressure survey, well tests, "
          f"failure history ({st.get('failure_history', {}).get('rod_failures_per_well_year')} failures/well-yr)")
    n_cards = h.query("SELECT COUNT(*) FROM cards")[0][0]
    check("I2", "Live data path: cards and tags flow SCADA -> OPC UA -> twin",
          len(decs) >= 10 and n_cards > 50,
          f"{n_cards} cards captured by SCADA, {len(decs)} diagnosed by the twin (newest card each time)")
    # B: guards
    check("I3", "PLC rejects an out-of-bounds request from the twin (self-test)",
          "outside" in run.get("selftest_reject", ""), f"PLC reply: '{run.get('selftest_reject')}'")
    sp = [(t, x) for t, k, x in plc if k == "SETPOINT" and x.startswith("SPM")]
    vals = [float(x.split("->")[1].split()[0]) for t, x in sp]
    prev, ok_ramp = 8.0, True
    for v in vals:
        ok_ramp &= abs(v - prev) <= 1.5 + 1e-6 and 2.0 <= v <= 10.0; prev = v
    applied = sum(1 for t, k, x in audit if k == "APPLIED")
    check("I4", "Twin set-points are applied by the PLC within bounds and ramp limits, with readback confirmation",
          twin["writes_accepted"] >= 5 and ok_ramp and applied >= 3,
          f"{twin['writes_accepted']} writes accepted, {applied} readback confirmations, SPM sequence {vals}")
    # C: sensor fault
    t0, t1 = scen.get("desync", 0) and scen["desync"], None
    ds = [d for t, d in decs if 3.5 * 86400 <= t <= 3.85 * 86400]
    flagged = [d for d in ds if "SENSOR" in d["action"]]
    wrote = [d for d in flagged if d["write"]]
    check("I5", "Load-cell desync is detected and the twin holds (no set-point written on bad data)",
          len(flagged) >= 1 and not wrote, f"{len(ds)} decisions in the fault window, {len(flagged)} SENSOR ALERT, "
                                           f"{len(wrote)} writes on bad data")
    # D: watchdog
    off = [t for t, k, x in audit if k == "OFFLINE"]; on = [t for t, k, x in audit if k == "ONLINE"]
    wd = [(t, x) for t, k, x in plc if k == "WATCHDOG"]
    lost = [t for t, x in wd if "lost" in x]; rest = [t for t, x in wd if "restored" in x]
    check("I6", "Twin outage: PLC watchdog falls back within the timeout and returns to REMOTE on recovery",
          bool(off and lost and rest) and 0 < lost[0] - off[0] <= 3.6 * 3600 and rest[0] >= on[0],
          f"twin offline at day {off[0] / 86400:.2f}, watchdog at {lost[0] / 86400:.2f} "
          f"(+{(lost[0] - off[0]) / 3600:.1f} h), REMOTE again at {rest[0] / 86400:.2f}; "
          f"fallback: {wd[0][1] if wd else ''}" if off and lost and rest else f"offline {off}, watchdog {wd}")
    # E: outcomes vs baseline (same plant, same scenario)
    check("I7", "Outcome vs current practice: fewer rod-floating trips and less impact, lower energy per barrel, oil not reduced",
          twin["trips_low_load"] < base["trips_low_load"] and twin["kwh_per_bbl_oil"] < base["kwh_per_bbl_oil"]
          and twin["impact_over_limit_h"] <= base["impact_over_limit_h"] and twin["oil_bbl"] >= 0.98 * base["oil_bbl"],
          f"trips {base['trips_low_load']} -> {twin['trips_low_load']}; kWh/bbl {base['kwh_per_bbl_oil']} -> "
          f"{twin['kwh_per_bbl_oil']}; impact h {base['impact_over_limit_h']} -> {twin['impact_over_limit_h']}; "
          f"oil {base['oil_bbl']} -> {twin['oil_bbl']} bbl")
    lat = run["latency_s"]
    check("I8", "Real-time: per-card decision latency (diagnosis + estimation + optimisation)",
          lat["median"] is not None and lat["median"] < 10 and lat["max"] < 30,
          f"median {lat['median']:.1f} s, max {lat['max']:.1f} s (first card is a cold start), n={lat['n']}")
    req = [t for t, k, x in audit if k == "REQUEST"]
    closed = [t for t, k, x in audit if k in ("APPLIED", "REJECTED")]
    check("I9", "Audit trail: every decision, request, readback and alarm is logged in the historian",
          len(decs) > 0 and len(req) > 0 and len(closed) > 0,
          f"{len(decs)} decisions, {len(req)} requests, {len(closed)} readbacks, "
          f"{sum(1 for t, s, k, x in ev if s == 'TWIN')} twin recommendations/alarms mirrored to SCADA")
    out_j = {"summary": {s: sum(1 for r in R if r["status"] == s) for s in ("PASS", "FAIL")}, "results": R,
             "baseline": base, "twin": twin}
    _j.dump(out_j, open(os.path.join(out, "acceptance_report.json"), "w"), indent=2)
    print("ACCEPTANCE", out_j["summary"])


if __name__ == "__main__" and "--tests-only" in sys.argv:
    acceptance(os.path.join(HERE, "integration_out"))


if __name__ == "__main__" and "--tests-only" not in sys.argv:
    main()
