"""
Seed the historian with a synthetic 3-year history for a 10-well Baghewala-like field.

Covers every data type the problem statement lists: production history, CSS cycle records,
steam injection parameters, VFD & SRP operating data, rod-failure and pump-unsetting
history, well completion data, fluid properties and pressure data.

Practice is the 'historical' one: fixed 12/2/~180-day cycles and fixed 8 SPM with manual
changes -- so failures cluster where the physics says they should (cold, viscous oil at
high SPM -> floating/buckling; low fillage -> pounding/unsetting).
"""
import datetime as dt
import json
import math

import numpy as np
import torch

from .config import COMPLETION, TRUE_WELL
from .plant import P

WELLS = [f"BGW-{i:02d}" for i in range(1, 11)]


def _well_params(rng, i):
    if i == 0:                                   # BGW-01 = the live well (true parameters)
        w = TRUE_WELL["BGW-01"]
        return dict(h=w["h"], k_md=w["k_md"], mu_cold=w["mu_cold"], mu_hot=w["mu_hot"], T0=w["T0"],
                    Ts=w["Ts"], api=w["api"], asph=w["asphaltene_wt_pct"], rh=21.0, rc=70.0, J_mult=4.6)
    return dict(h=rng.uniform(15, 25), k_md=float(np.exp(rng.uniform(np.log(100), np.log(250)))),
                mu_cold=float(np.exp(rng.uniform(np.log(8), np.log(15)))),
                mu_hot=float(np.exp(rng.uniform(np.log(0.2), np.log(0.5)))), T0=rng.uniform(319.2, 321.2),
                Ts=523.0, api=rng.uniform(17, 19), asph=rng.uniform(7, 11), rh=rng.uniform(17, 24),
                rc=rng.uniform(60, 80), J_mult=rng.uniform(3.8, 5.4))


def _daily_oil(w, tau_days, dp):
    tau = torch.tensor(tau_days, dtype=torch.float32).unsqueeze(1)
    p = {k: torch.full_like(tau, float(w[k])) for k in ["h", "k_md", "mu_cold", "mu_hot", "T0", "Ts", "rh", "rc", "J_mult"]}
    p["alpha"] = torch.full_like(tau, P.ALPHA0)
    th = P.theta_boberg_lantz(tau, p["rh"], p["h"], p["alpha"])
    T = p["T0"] + th * (p["Ts"] - p["T0"])
    mu = P.andrade_mu(T, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"])
    q = P.dupuy_J_zonal(mu, p) * dp * P.M3S_TO_BPD
    T_tub = torch.clamp(305.0 + 0.6 * (T - 305.0), min=p["T0"])            # tubing (60 % of reservoir excess)
    mu_tub = P.andrade_mu(T_tub, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"])
    f = lambda x: x.squeeze(1).double().numpy()
    return f(q), f(T), f(mu_tub)


def seed(hist, years=3, seed=1, start=dt.date(2023, 1, 1)):
    rng = np.random.default_rng(seed)
    dp = (1378.0 - 87.0) * 6894.757                     # reservoir minus pumped-off intake
    counts = {"production_daily": 0, "css_cycles": 0, "rod_failures": 0, "pump_unsetting": 0}
    for i, well in enumerate(WELLS):
        w = _well_params(rng, i)
        comp = dict(COMPLETION["BGW-01"]); comp["pump_depth_m"] = float(rng.uniform(1030, 1080)) if i else 1050.0
        hist.db.execute("INSERT OR REPLACE INTO completion VALUES (?,?)", (well, json.dumps(comp)))
        visc_table = {f"{Tc}C": round(float(P.andrade_mu(torch.tensor(Tc + 273.15), torch.tensor(w["mu_cold"]),
                                                          torch.tensor(w["mu_hot"]), torch.tensor(w["T0"]),
                                                          torch.tensor(w["Ts"]))) * 1000, 1) for Tc in (50, 100, 150, 200, 250)}
        hist.db.execute("INSERT OR REPLACE INTO fluid_properties VALUES (?,?,?,?,?)",
                        (well, round(w["api"], 1), visc_table["50C"], round(w["asph"], 1), json.dumps(visc_table)))
        day, cycle, n_days = 0, 1, int(365 * years)
        spm = 8.0
        while day < n_days:
            inj, soak = float(rng.uniform(10, 14)), 2.0
            prod = float(rng.uniform(160, 200))
            rate_tpd = float(rng.uniform(130, 160)); press = float(rng.uniform(1300, 1650)); qual = float(rng.uniform(0.72, 0.8))
            d0 = start + dt.timedelta(days=day)
            for k in range(int(inj)):
                hist.db.execute("INSERT INTO steam_injection_daily VALUES (?,?,?,?,?,?)",
                                ((d0 + dt.timedelta(days=k)).isoformat(), well, rate_tpd * rng.normal(1, 0.03),
                                 press * rng.normal(1, 0.02), qual, rate_tpd * 0.075 * rng.normal(1, 0.03)))
            tau = np.arange(int(prod)) + soak
            q, T, mu = _daily_oil(w, tau, dp)                   # mu = tubing viscosity
            steam_cwe = rate_tpd * inj * 6.28981
            water = 0.45 * steam_cwe / 25 * np.exp(-np.arange(int(prod)) / 25)
            cyc_oil, down = 0.0, 0
            for k in range(int(prod)):
                date = start + dt.timedelta(days=day + int(inj + soak) + k)
                if day + inj + soak + k >= n_days:
                    break
                T_wh = float(32 + 0.35 * (T[k] - 305))
                if down > 0:
                    down -= 1
                    hist.db.execute("INSERT INTO production_daily VALUES (?,?,?,?,?,?)",
                                    (date.isoformat(), well, 0.0, 0.0, 0.0, "workover"))
                    continue
                # manual, reactive SPM practice: raise when oil is hot, cut after problems
                liquid = float(q[k] + water[k])
                fill = min(1.0, liquid / (0.1166 * 2.25 * 80 * spm * 0.95))
                run = min(1.0, max(0.3, fill / 0.8))
                oil = q[k] * rng.normal(1, 0.06) * min(1.0, run / max(fill / 0.8, 1e-6))
                hist.db.execute("INSERT INTO production_daily VALUES (?,?,?,?,?,?)",
                                (date.isoformat(), well, float(max(oil, 0)), float(water[k] * rng.normal(1, 0.06)),
                                 float(run), ""))
                counts["production_daily"] += 1
                kwh = float(24 * run * (6 + 0.35 * spm * (1 + mu[k] * 2)))
                hist.db.execute("INSERT INTO vfd_daily VALUES (?,?,?,?,?,?,?,?)",
                                (date.isoformat(), well, spm * 60 / 8, spm, 86.0, kwh * rng.normal(1, 0.04), run,
                                 kwh / 24 / max(run, 0.1) * 1000 / (1.732 * 415 * 0.85)))
                if k % 7 == 3:
                    hist.db.execute("INSERT INTO well_tests VALUES (?,?,?,?,?,?)",
                                    (date.isoformat(), well, float(oil * rng.normal(1, 0.04)),
                                     float(water[k] * rng.normal(1, 0.05)), T_wh, 24.0))
                cyc_oil += max(oil, 0)
                # failure hazards (per day): viscous oil + high SPM -> rod failure; low fillage -> unsetting
                mu_tub = float(mu[k])                                    # tubing oil viscosity, Pa.s
                p_rod = 1.5e-4 * math.exp(0.6 * (spm - 6)) * (1 + 10 * min(mu_tub / 1.5, 4) ** 2)
                p_uns = 1.2e-4 * (1 + 25 * max(0.0, 0.7 - fill))
                if rng.random() < p_rod:
                    mode = "rod part / buckling (compressive, floating)" if mu_tub > 1.5 else "fatigue part"
                    hist.db.execute("INSERT INTO rod_failures VALUES (?,?,?,?,?,?,?)",
                                    (date.isoformat(), well, float(rng.uniform(300, comp["pump_depth_m"] / 0.3048)), mode,
                                     spm, T_wh, 5.0))
                    counts["rod_failures"] += 1; down = 5; spm = max(4.0, spm - 1)
                elif rng.random() < p_uns:
                    hist.db.execute("INSERT INTO pump_unsetting VALUES (?,?,?,?,?)",
                                    (date.isoformat(), well, float(max(0, 30 * (0.7 - fill))), spm, 4.0))
                    counts["pump_unsetting"] += 1; down = 4
                if k == 0:
                    spm = 8.0                      # crews run fast after each steam cycle
            hist.db.execute("INSERT INTO css_cycles VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (well, cycle, d0.isoformat(), inj, soak, prod, rate_tpd * inj, press, qual, float(cyc_oil)))
            counts["css_cycles"] += 1
            day += int(inj + soak + prod); cycle += 1
        for y in range(years):
            hist.db.execute("INSERT INTO pressure_surveys VALUES (?,?,?,?)",
                            (dt.date(start.year + y, 6, 1).isoformat(), well, 1378.0 * (1 - 0.01 * y) * rng.normal(1, 0.01), 1050.0))
    hist.commit()
    return counts
