"""
True-well plant model for the synthetic SCADA environment.

This is the "real well" the twin must discover. It shares textbook physics with the
twin (Boberg-Lantz cooling, zonal Dupuy inflow, Everitt-Jennings rod wave equation) but
uses DIFFERENT, hidden parameters, adds dynamics the twin does not model (annulus fluid
level storage), sensor noise, and injectable faults.
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "twin"))
import pinn_digital_twin as P  # noqa: E402  (physics functions only; no trained models used)

from .config import COMPLETION, TRUE_WELL, CARD_INTERVAL_S, annulus_area_m2  # noqa: E402

PSI = 6894.757
T_AMB = 305.0


class TrueWell:
    def __init__(self, well_id, start_production_day=120.0, seed=0):
        self.id = well_id
        self.c = dict(COMPLETION[well_id])
        self.w = dict(TRUE_WELL[well_id])
        self.rng = np.random.default_rng(seed)
        self.A_ann = annulus_area_m2(self.c)
        self.L_ft = self.c["pump_depth_m"] / 0.3048
        self.t = 0.0                                   # simulated seconds since start
        self.prod_day0 = start_production_day
        # hidden reservoir parameters (true, not the twin's)
        self.rh, self.rc, self.J_mult = 21.0, 70.0, 4.6
        self.steam_cwe_bbl = 11000.0
        # pump state
        self.spm, self.stroke, self.run_frac = 8.0, 86.0, 1.0
        self.running, self.idle_left, self.trip_left = True, 0.0, 0.0
        self.V = 0.5 * self.A_ann * 300.0              # annulus liquid volume above pump, m3
        self.fill_hist = []
        self.duty_clock = 0.0
        # faults / scenario knobs
        self.load_gain, self.desync = 1.0, False
        # accounting
        self.acc = {"oil_bbl": 0.0, "water_bbl": 0.0, "kwh": 0.0, "run_s": 0.0, "impact_over_h": 0.0,
                    "strokes": 0.0}
        self.last_card = None
        self.last_card_t = -1e9
        self.card_id = 0
        self.motor_kw = 0.0
        self.card_work = 0.0

    # ------------------------------------------------------------------ reservoir
    def days_since_injection_end(self):
        return self.w["t_soak"] + self.prod_day0 + self.t / 86400.0

    def reservoir_state(self):
        tau = torch.tensor([[self.days_since_injection_end()]])
        p = {k: torch.tensor([[float(self.w[k])]]) for k in ["h", "k_md", "mu_cold", "mu_hot", "T0", "Ts"]}
        p["alpha"] = torch.tensor([[P.ALPHA0]]); p["rh"] = torch.tensor([[self.rh]])
        p["rc"] = torch.tensor([[self.rc]]); p["J_mult"] = torch.tensor([[self.J_mult]])
        th = P.theta_boberg_lantz(tau, p["rh"], p["h"], p["alpha"])
        T = p["T0"] + th * (p["Ts"] - p["T0"])
        mu = P.andrade_mu(T, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"])
        J = P.dupuy_J_zonal(mu, p) * self.w["productivity_scale"]            # m3/s/Pa
        return float(T), float(J)

    def water_rate_m3s(self):
        d = self.prod_day0 + self.t / 86400.0
        q_bpd = 0.45 * self.steam_cwe_bbl / 25.0 * math.exp(-d / 25.0)
        return q_bpd * 0.158987 / 86400.0

    def tubing_viscosity(self, T_res, wc):
        T_tub = max(T_AMB + self.w["tubing_temp_fraction"] * (T_res - T_AMB), self.w["T0"])
        mu_o = float(P.andrade_mu(torch.tensor(T_tub), torch.tensor(self.w["mu_cold"]), torch.tensor(self.w["mu_hot"]),
                                  torch.tensor(self.w["T0"]), torch.tensor(self.w["Ts"])))
        return mu_o ** (1 - wc) * 0.0005 ** wc, T_tub

    # ------------------------------------------------------------------ one plant step
    def step(self, dt, plc):
        T_res, J = self.reservoir_state()
        pr = self.w["p_reservoir_psi"] * PSI
        p_cas = self.w["casing_pressure_psi"] * PSI
        rho = 960.0
        h_f = self.V / self.A_ann
        pw = p_cas + rho * 9.81 * h_f
        q_oil_in = max(J * (pr - pw), 0.0)
        q_w_in = self.water_rate_m3s()
        q_in = q_oil_in + q_w_in
        wc = q_w_in / max(q_in, 1e-12)
        # pump availability: trips, pump-off idle, twin run-fraction duty cycle
        self.duty_clock = (self.duty_clock + dt) % 3600.0
        duty_on = self.duty_clock < 3600.0 * self.run_frac
        if self.trip_left > 0:
            self.trip_left -= dt; self.running = False
        elif self.idle_left > 0:
            self.idle_left -= dt; self.running = False
        else:
            self.running = duty_on
        pumped, fill = 0.0, None
        if self.running:
            Sp = self.stroke * 0.93
            PD = 0.1166 * self.c["plunger_in"] ** 2 * Sp * self.spm * 0.95 * 0.158987 / 86400.0   # m3/s
            avail = self.V + q_in * dt
            pumped = min(PD * dt, avail)
            fill = pumped / max(PD * dt, 1e-12)
            self.V = avail - pumped
            self.fill_hist.append(fill); self.fill_hist = self.fill_hist[-10:]
            self.acc["run_s"] += dt
            self.acc["strokes"] += self.spm * dt / 60.0
            # pump-off controller: idle when fillage stays low
            if len(self.fill_hist) >= 3 and np.mean(self.fill_hist[-3:]) < plc.cfg["poc_fillage_sp"]:
                self.idle_left = plc.cfg["poc_idle_min"] * 60.0
                self.fill_hist = []
            # impact at plunger/fluid contact (true pump), time over the allowance
            if fill < 0.999:
                phi = math.acos(max(-1.0, min(1.0, 1 - 2 * (1 - fill))))
                v_imp = (Sp / 12 / 2) * (2 * math.pi * self.spm / 60) * math.sin(phi)
                A = math.pi / 4 * self.c["rod_diam_in"] ** 2
                v_w = math.sqrt(144 * self.c["rod_E_psi"] * 32.2 / self.c["rod_density_lbm_ft3"])
                if self.c["rod_E_psi"] * A / v_w * v_imp > 1200.0:
                    self.acc["impact_over_h"] += dt / 3600.0
        else:
            self.V += q_in * dt
        self.V = max(self.V, 0.0)
        frac_oil = 1 - wc
        self.acc["oil_bbl"] += pumped * frac_oil / 0.158987
        self.acc["water_bbl"] += pumped * wc / 0.158987
        power = self.motor_kw if self.running else 0.3
        self.acc["kwh"] += power * dt / 3600.0
        self.t += dt
        self.state = {"T_res": T_res, "q_oil_in_bpd": q_oil_in * 86400 / 0.158987, "wc": wc,
                      "h_f_ft": h_f / 0.3048, "pw_psi": pw / PSI, "fill": fill}
        # dynamometer card
        if self.running and self.t - self.last_card_t >= CARD_INTERVAL_S:
            self.capture_card(T_res, wc, h_f)
        return self.state

    # ------------------------------------------------------------------ card capture
    def capture_card(self, T_res, wc, h_f):
        mu_t, _ = self.tubing_viscosity(T_res, wc)
        rod = dict(diam=self.c["rod_diam_in"], L=self.L_ft, E=self.c["rod_E_psi"], rho=self.c["rod_density_lbm_ft3"],
                   spm=self.spm, stroke=self.stroke, sg=0.96)
        damp = P.damping_from_viscosity(mu_t, rod["diam"], self.c["tubing_id_in"], rod["rho"]) * self.w["rod_friction_scale"]
        fill = float(np.mean(self.fill_hist)) if self.fill_hist else 1.0
        net_lift = max(self.L_ft - h_f / 0.3048, 0.0)
        W_f = 0.340 * 0.96 * self.c["plunger_in"] ** 2 * net_lift
        card = P.simulate_dynamometer_card(rod, min(fill, 1.0), damp, fluid_load_lbf=W_f,
                                           plunger_in=self.c["plunger_in"], n_out=400)
        pos = card.surface_position_ft.values.copy()
        load = card.surface_load_lbf.values.copy()
        self.card_work = P.card_work(pos, load)                     # true (sensor-independent) work
        prhp_w = max(self.card_work, 0.0) * self.spm / 60.0 * 1.3558
        self.motor_kw = prhp_w / (0.85 * 0.90) / 1000.0 + 0.4
        # sensors: noise, load-cell gain error, channel desynchronisation
        load = load * self.load_gain + self.rng.normal(0, 25.0, len(load))
        pos = pos + self.rng.normal(0, 0.005, len(pos))
        if self.desync:
            load = np.roll(load, len(load) // 4)
        self.last_card = {"t": card.t_sec_within_stroke.values, "pos": pos, "load": load,
                          "true_fill": fill, "true_mu": mu_t, "true_damping": damp}
        self.last_card_t = self.t
        self.card_id += 1

    def static_rod_weight(self):
        A = math.pi / 4 * self.c["rod_diam_in"] ** 2
        return self.c["rod_density_lbm_ft3"] * A / 144 * self.L_ft * (1 - 62.4 * 0.96 / self.c["rod_density_lbm_ft3"])
