"""
=============================================================================
  PINN Well-to-Surface Digital Twin -- CSS + SRP  (Baghewala heavy-oil field)
  v2: corrected + compact (< 0.1 M trainable parameters in total)
=============================================================================

WHAT THE TWIN DOES (mapped to the problem statement)
  1. Reservoir PINN (~51k params)
       Predicts reservoir HEATING (heated-zone growth vs injected steam),
       COOLING (Boberg-Lantz average temperature) and PRODUCTION (Dupuy zonal
       inflow with Andrade viscosity) as a smooth, differentiable function of
       time AND of the controllable CSS parameters (steam volume, soak time).
  2. SRP PINN (~46k params)
       Everitt & Jennings damped-wave equation solved as a PINN: measured
       SURFACE card (position + load) -> DOWNHOLE pump card. Rod damping is a
       set parameter (E&J); it can be learned (--learn-damping) only when a
       downhole reference is available, because surface data alone do not
       identify it.  Feeds card
       diagnostics: rod floating, fluid pound / impact loading, fillage.
  3. CSS cycle optimizer
       Gradient-based optimisation THROUGH the reservoir PINN of injection
       days (steam volume), soak days and production days -> max profit/day,
       reports SOR, energy per bbl, cost per bbl. Verified against the
       analytical physics. Multi-cycle plan uses the cycle-to-cycle OSR
       decline calibrated on the REAL field data (Basta et al. 2021).
  4. SRP operation optimizer (runs every SCADA tick)
       Chooses SPM (-> VFD frequency) and stroke length from the current
       well state (reservoir PINN rate + temperature -> oil viscosity) with
       physics constraints: no rod floating (viscous drag vs buoyed rod
       weight), bounded impact load (fluid pound at plunger/fluid contact),
       modified-Goodman rod stress, gearbox torque, structure load, pump
       fillage.  Compared day-by-day against a fixed-speed baseline.
  5. Field-data hooks: production history / CSS records / steam injection /
       VFD-SRP cards (FIELD_SCHEMA + fine_tune_reservoir + diagnose_scada_card,
       with simulate_dynamometer_card as an independent forward model for testing).

CORRECTIONS vs v1 (see CHANGES in the chat for details)
  * Runs on the CSVs actually shipped (per-well / per-case parameters come from
    the generators' registries, not from missing CSV columns).
  * All inputs and PDE residuals non-dimensionalised; no Fourier features on
    raw SI values; float32 by default.
  * Removed the uncertainty-weighting layer that diverged on zero losses.
  * Reservoir physics consistent with the lumped (average-T) data: Boberg-Lantz
    exact closed form, cooling monotonicity, Dupuy rate with a bounded,
    regularised correction term (so real history can pull it).
  * SRP: exact surface-position BC (Fourier fit, as in Gibbs' method), correct
    time units, per-case rod properties, surface LOAD as Cauchy data, pump card
    predicted (not free-end forced), correct Archimedes buoyancy.
  * CSS optimizer gradients actually flow through the reservoir PINN.
  * Coupling reservoir -> SRP is real: T -> viscosity -> rod drag -> floating
    limit on SPM; PINN productivity -> inflow/outflow match -> fillage.
  * Inference no longer crashes (autograd only where needed).

Values tagged CAL=ASSUMED are engineering assumptions to be replaced by
Baghewala field data (see FIELD_CONFIG).  The synthetic training data give
correct curve SHAPES; absolute magnitudes need calibration with real
production history (fine_tune_reservoir).
=============================================================================
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy 1.x / 2.x

# =============================================================================
# 0. REGISTRIES (must match the generator scripts) and FIELD CONFIG
# =============================================================================
ROCK = dict(rho_rock=2200.0, c_rock=500.0, lam=0.655, phi=0.20,
            rho_steam=50.0, c_steam=4043.0, latent=2.3e6,
            rw=0.025, rc=200.0, pr=15.0e6, pw=12.5e6)       # generator (Gilmanov) base
DP_REF = ROCK["pr"] - ROCK["pw"]                             # drawdown used in synthetic data
M3S_TO_BPD = 543439.7

WELL_REGISTRY = {
    # t_inj / t_soak are the generator's cycle schedule; Qs = steam rate (m3/s @ 50 kg/m3)
    "GILMANOV_REF_WELL": dict(h=30.0, k_md=200.0, mu_cold=0.005, mu_hot=0.7, T0=300.0,
                              Ts=610.0, t_inj=10.54, t_soak=1.13, Qs=4.96e-3),
    "BGW_analogue_A": dict(h=20.0, k_md=150.0, mu_cold=10.0, mu_hot=0.30, T0=320.0,
                           Ts=610.0, t_inj=12.0, t_soak=2.0, Qs=4.96e-3),
    "BGW_analogue_B": dict(h=25.0, k_md=250.0, mu_cold=8.0, mu_hot=0.20, T0=319.0,
                           Ts=610.0, t_inj=12.0, t_soak=2.0, Qs=4.96e-3),
    "BGW_analogue_C": dict(h=15.0, k_md=100.0, mu_cold=15.0, mu_hot=0.50, T0=321.0,
                           Ts=610.0, t_inj=12.0, t_soak=2.0, Qs=4.96e-3),
}

# -----------------------------------------------------------------------------
# BAGHEWALA FIELD (target of the twin).  Sources: problem statement (46-48 C,
# 17-19 API, high viscosity, low pressure); SPE-23APOG / OIL India field page as
# quoted in the dataset README (8,000-15,000 cP @ 50 C, Jodhpur Sandstone,
# ~1,100-1,150 m).  h, k, mu at steam temperature: CAL=ASSUMED, spanned by the
# three BGW analogue wells in the training data.
# -----------------------------------------------------------------------------
BAGHEWALA_RANGES = {            # (low, P50, high, sampling)
    "T0": (319.15, 320.15, 321.15, "lin"),     # 46-48 C
    "mu_cold": (8.0, 11.0, 15.0, "log"),       # Pa.s
    "mu_hot": (0.2, 0.3, 0.5, "log"),          # Pa.s at steam T   CAL=ASSUMED
    "k_md": (100.0, 150.0, 250.0, "log"),      # CAL=ASSUMED
    "h": (15.0, 20.0, 25.0, "lin"),            # m  CAL=ASSUMED
    "Ts": (610.0, 610.0, 620.0, "lin"),        # K
}
WELL_REGISTRY["BAGHEWALA_P50"] = dict({k: v[1] for k, v in BAGHEWALA_RANGES.items()},
                                      t_inj=12.0, t_soak=2.0, Qs=4.96e-3)


def sample_baghewala_wells(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = {}
        for k, (lo, _, hi, kind) in BAGHEWALA_RANGES.items():
            w[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if kind == "log"
                         else rng.uniform(lo, hi))
        cal = WELL_REGISTRY["BAGHEWALA_P50"]
        out.append(dict(w, t_inj=12.0, t_soak=2.0, Qs=cal["Qs"],
                        rc=cal.get("rc", ROCK["rc"]), J_mult=cal.get("J_mult", 1.0)))
    return out


# -----------------------------------------------------------------------------
# PUBLIC FIELD-LEVEL TARGETS for calibration (no well-level Baghewala data is public)
# -----------------------------------------------------------------------------
FIELD_TARGETS = {
    "rate_per_well_bpd": {"low": 17.0, "mid": 26.0, "high": 36.0,
                          "source": "Oil India: >600 bpd from 35 producing wells (oil-india.com/rajasthan-fields); "
                                    "record 1,202 bpd from 33 producing wells, FY2025-26 (press, 2026)"},
    "steam_uplift_x": {"low": 5.0, "mid": 5.5, "high": 6.0,
                       "source": "Oil India first CSS pilot: 5-6 fold production increase after steam injection"},
    "cycle_OSR": {"low": 0.19, "mid": 0.43, "high": 1.0,
                  "source": "Sheng, Enhanced Oil Recovery Field Case Studies, Ch.16 (CSS): P50 OSR of actual "
                            "CSS field data = 0.43; case histories 0.19-1.0. ANALOGUE statistic, not Baghewala."},
}


def _oracle_rates(wp, cfg, sched, days=None):
    """Physics (Boberg-Lantz + zonal Dupuy) daily oil rate at field drawdown and the cold rate."""
    days = days or int(sched["t_prod"])
    t = torch.arange(0, days, 1.0).unsqueeze(1)
    p = well_params_tensor(wp, len(t), "cpu", wp["Qs"] * sched["t_inj"] * 86400.0)
    th = theta_boberg_lantz(t + sched["t_soak"], p["rh"], p["h"], p["alpha"])
    T = p["T0"] + th * (p["Ts"] - p["T0"])
    dp = cfg["p_reservoir_Pa"] - cfg["pw_min_Pa"]
    q = (dupuy_J_zonal(andrade_mu(T, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"]), p) * dp * M3S_TO_BPD).squeeze(1)
    q_cold = float(dupuy_J_cold(p)[0] * dp * M3S_TO_BPD)
    return q.numpy(), q_cold, float(p["rh"][0])


def calibrate_baghewala(cfg, sched=None, n_outer=8):
    """Three CAL=ASSUMED quantities are fitted to three public numbers (joint fixed point):
       * drainage radius rc   <- steam uplift, first-30-day rate / cold rate = 5.5x
         (independent of the productivity scale);
       * productivity multiplier J_mult <- average production-phase rate 26 bbl/d/well;
       * steam injection rate Qs <- cycle oil-steam ratio 0.43 (analogue P50).
       Steam volume changes the heated radius, hence the uplift fit, so the three are
       iterated to convergence. Literature values (viscosity, T0, depth, API) untouched.
       Updates BAGHEWALA_P50 in place and returns a report."""
    wp0 = dict(WELL_REGISTRY["BAGHEWALA_P50"])
    wp = dict(wp0, rc=ROCK["rc"], J_mult=1.0, Qs=4.96e-3)
    sched = sched or {"t_inj": wp["t_inj"], "t_soak": wp["t_soak"], "t_prod": 180.0}
    before_q, before_c, _ = _oracle_rates(wp, cfg, sched)
    steam_cwe = lambda Qs: Qs * sched["t_inj"] * 86400.0 * ROCK["rho_steam"] / 1000.0 * 6.28981
    before_osr = before_q.sum() / steam_cwe(wp["Qs"])
    t_up, t_rate, t_osr = (FIELD_TARGETS[k]["mid"] for k in ["steam_uplift_x", "rate_per_well_bpd", "cycle_OSR"])
    for _ in range(n_outer):
        _, _, rh = _oracle_rates(wp, cfg, sched)

        def uplift(rc):
            q, qc, _ = _oracle_rates(dict(wp, rc=rc, J_mult=1.0), cfg, sched)
            return q[:30].mean() / qc
        lo, hi = rh * 1.05, 5000.0
        if uplift(lo) < t_up:
            raise ValueError("uplift target not reachable with drainage radius alone")
        for _ in range(50):                       # uplift decreases monotonically with rc
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if uplift(mid) > t_up else (lo, mid)
        wp["rc"] = math.sqrt(lo * hi)
        q1, _, _ = _oracle_rates(dict(wp, J_mult=1.0), cfg, sched)
        wp["J_mult"] = float(t_rate / q1.mean())
        oil = t_rate * sched["t_prod"]
        wp["Qs"] = float(wp["Qs"] * (oil / t_osr) / steam_cwe(wp["Qs"]))   # steam volume for target OSR
    WELL_REGISTRY["BAGHEWALA_P50"].update(rc=wp["rc"], J_mult=wp["J_mult"], Qs=wp["Qs"])
    q2, qc2, rh2 = _oracle_rates(WELL_REGISTRY["BAGHEWALA_P50"], cfg, sched)
    steam_t_day = wp["Qs"] * ROCK["rho_steam"] * 86400 / 1000
    return {"method": "joint fixed point: rc <- uplift, J_mult <- rate, steam rate <- OSR (physics oracle)",
            "schedule_assumed_for_targets": sched, "targets": FIELD_TARGETS,
            "calibrated": {"drainage_radius_m": round(wp["rc"], 2), "productivity_multiplier": round(wp["J_mult"], 3),
                           "steam_rate_t_per_day": round(steam_t_day, 1),
                           "steam_per_cycle_bbl_cwe": round(steam_cwe(wp["Qs"]), 0),
                           "heated_radius_m": round(rh2, 2)},
            "before": {"avg_rate_bpd": round(float(before_q.mean()), 2),
                       "uplift_x": round(float(before_q[:30].mean() / before_c), 2),
                       "cycle_OSR": round(float(before_osr), 3), "steam_rate_t_per_day": 21.4},
            "after": {"avg_rate_bpd": round(float(q2.mean()), 2), "uplift_x": round(float(q2[:30].mean() / qc2), 2),
                      "cycle_OSR": round(float(q2.sum() / steam_cwe(wp["Qs"])), 3), "cold_rate_bpd": round(qc2, 2)},
            "caveats": ["Calibrated to FIELD AVERAGES, not to well histories: three numbers fix three "
                        "parameters, so there is no independent validation of the calibration.",
                        "The OSR target is an analogue statistic (Sheng Ch.16), not a Baghewala figure.",
                        "Uplift could equally come from near-wellbore skin removal; drainage radius and skin "
                        "are not separable with these data.",
                        "Field average includes later cycles and different schedules; the fit assumes the "
                        "current 12/2/180 schedule."]}


SRP_CASES = {   # Everitt & Jennings (1992) Tables 1-4, as in gen_srp_dynamometer_dataset.py
    "Case1_Lufkin_C114": dict(diam=0.75, L=2000, E=30.5e6, rho=490.0, spm=15, stroke=54, sg=1.0),
    "Case2_American_C228": dict(diam=1.25, L=3000, E=8.03e6, rho=95.0, spm=10, stroke=100, sg=1.0),
    "Case3_Lufkin_M228": dict(diam=0.875, L=1500, E=30.5e6, rho=490.0, spm=8, stroke=121, sg=0.92),
    "Case4_Lufkin_C114b": dict(diam=0.75, L=3179, E=30.5e6, rho=490.0, spm=10, stroke=64, sg=0.8),
}
G_C = 32.2
SRP_DAMPING_PER_S = 1.0   # E&J viscous damping; synthetic generator value. CAL: set from E&J energy balance

FIELD_CONFIG = {
    # --- reservoir / well (Baghewala, Jodhpur Sandstone) ---
    "p_reservoir_Pa": 11.37e6,    # CAL=ASSUMED near-hydrostatic at ~1,050 m ("low pressure")
    "pw_min_Pa": 0.6e6,           # pumped-off pump-intake pressure (CAL=ASSUMED)
    "p_casing_Pa": 0.3e6,         # CAL=ASSUMED
    "pump_depth_m": 1050.0,
    "oil_sg": 0.947,              # 18 API (problem statement: 17-19 API)
    "water_visc_Pa_s": 0.0005,
    "T_ambient_K": 305.0,
    "tubing_temp_fraction": 0.7,  # T_tubing = T_amb + f (T_res - T_amb)  CAL=ASSUMED
    "condensate_return_frac": 0.45,   # fraction of injected water produced back  CAL=ASSUMED
    "condensate_decay_days": 25.0,    # CAL=ASSUMED; replace with measured water cut
    # --- SRP equipment (C-228D-213-86 class unit, grade D steel rods) ---
    "rod_diam_in": 0.875, "rod_weight_lb_ft": 2.22, "rod_E_psi": 30.5e6,
    "rod_tensile_psi": 115000.0, "goodman_SF": 0.9,
    "plunger_in": 1.5, "tubing_id_in": 2.441,
    "strokes_in": [54.0, 64.0, 74.0, 86.0],       # crank holes
    "spm_at_60Hz": 8.0, "vfd_Hz_min": 15.0, "vfd_Hz_max": 75.0,
    "structure_lbf": 21300.0, "gearbox_inlb": 228000.0,
    "holddown_lbf": 3000.0, "impact_allow_lbf": 1200.0,
    "eta_pump": 0.85, "eta_surface": 0.85, "eta_motor": 0.90, "slippage": 0.05,
    "fillage_target": 0.85, "rod_float_limit": 0.80, "goodman_limit": 0.90,
    # rod_float_limit 0.85 -> 0.80: the analytic (Mills) minimum load is ~23 % optimistic vs the
    # FD wave model (verify.py T8); 0.80 keeps the FD floating index <= ~0.85.
    # --- economics (USD) CAL=ASSUMED ---
    "oil_price": 65.0, "steam_cost_per_bbl_cwe": 4.5, "power_cost_kwh": 0.10,
    "opex_per_day": 120.0, "sor_limit": 6.0,
    # Steam availability: at these prices extra steam still pays for a single well, so the
    # binding limit is generator capacity shared across wells. CAL=ASSUMED: set from the
    # field's generator capacity / number of wells steamed per month.
    "max_injection_days": 25.0,
    # --- baseline practice for comparison ---
    "baseline_spm": 8.0, "baseline_stroke_in": 86.0,
}

# Column mapping for real Baghewala data (edit to your SCADA / historian export)
FIELD_SCHEMA = {
    "production_history": {"well_id": "well_id", "t_prod_days": "t_days_since_production_start",
                           "oil_rate_bpd": "oil_rate_bbl_day", "temp_K": "reservoir_temp_K"},
    "css_cycle_records": {"well_id": "well_id", "cycle": "cycle_no",
                          "cum_steam_bbl": "cumulative_steam_injection_bbl",
                          "cum_oil_bbl": "cumulative_oil_production_bbl"},
    "srp_card": {"t_s": "t_sec_within_stroke", "pos_ft": "surface_position_ft",
                 "load_lbf": "surface_load_lbf"},
}


# =============================================================================
# 1. PHYSICS (torch, differentiable)
# =============================================================================
def marx_langenheim_efficiency(t_inj_s, h):
    """Fraction of injected heat still in the steam zone after injecting for t_inj_s
    (Marx & Langenheim 1959): E_h = G(tD)/tD, G = e^tD erfc(sqrt tD) + 2 sqrt(tD/pi) - 1,
    tD = 4 K_ob M_ob t / (M_R^2 h^2). Heat is lost by conduction to cap and base rock."""
    M_R = (1 - ROCK["phi"]) * ROCK["rho_rock"] * ROCK["c_rock"]
    M_ob = ROCK["rho_rock"] * ROCK["c_rock"]
    tD = torch.clamp(torch.as_tensor(4 * ROCK["lam"] * M_ob * t_inj_s) / (M_R ** 2 * h ** 2), min=1e-9)
    G = torch.special.erfcx(torch.sqrt(tD)) + 2 * torch.sqrt(tD / math.pi) - 1
    return torch.clamp(G / tD, 0.0, 1.0)


def heated_radius(v_steam_m3, h, T0, Ts, t_inj_s=None):
    """Heated-zone radius from the injected-heat balance (heating phase). With t_inj_s,
    Marx-Langenheim heat losses to over/under-burden are applied (v3). Without it: the
    loss-free balance used by the data generator (kept for data QC)."""
    heat = v_steam_m3 * ROCK["rho_steam"] * (ROCK["c_steam"] * (Ts - T0) + ROCK["latent"])
    if t_inj_s is not None:
        heat = heat * marx_langenheim_efficiency(t_inj_s, h)
    cap = (1 - ROCK["phi"]) * ROCK["rho_rock"] * ROCK["c_rock"]
    return torch.sqrt(torch.clamp(heat / (math.pi * h * cap * (Ts - T0)), min=1e-6))


def theta_boberg_lantz(tau_days, rh, h, alpha):
    """Boberg-Lantz average dimensionless temperature (exact closed forms)."""
    t = tau_days * 86400.0
    tDr = torch.clamp(alpha * t / rh ** 2, min=1e-10)
    tDz = torch.clamp(alpha * t / (h / 2.0) ** 2, min=1e-10)
    x = 1.0 / (2.0 * tDr)
    T_Dr = 1.0 - (torch.special.i0e(x) + torch.special.i1e(x))
    T_Dz = torch.erf(1.0 / torch.sqrt(tDz)) - torch.sqrt(tDz / math.pi) * (1 - torch.exp(-1.0 / tDz))
    return torch.clamp(T_Dr, 0, 1) * torch.clamp(T_Dz, 0, 1)


def andrade_mu(T, mu_cold, mu_hot, T0, Ts):
    """ln(mu) = A + B/T through (T0, mu_cold) and (Ts, mu_hot)."""
    B = torch.log(mu_cold / mu_hot) / (1.0 / T0 - 1.0 / Ts)
    return mu_cold * torch.exp(B * (1.0 / T - 1.0 / T0))


def dupuy_J(mu_h, mu_c, rh, k_md, h, rc=None):
    """Zonal Dupuy productivity, m3/s per Pa."""
    k = k_md * 9.869233e-16
    rh = torch.clamp(rh, min=ROCK["rw"] * 1.01)
    rc = ROCK["rc"] if rc is None else rc
    rh = torch.minimum(rh, torch.as_tensor(rc) * 0.999) if torch.is_tensor(rc) else torch.clamp(rh, max=rc * 0.999)
    den = mu_h * torch.log(rh / ROCK["rw"]) + mu_c * torch.log(rc / rh)
    return 2 * math.pi * k * h / den


def dupuy_J_zonal(mu_h, p):
    """Productivity after stimulation: zonal (heated + cold) Dupuy, never below the
    unstimulated (cold) Dupuy -- the same rule the data generator and Gilmanov
    et al. (Eqs.18-19) use. Lets ALL wells (incl. Gilmanov's) share one model."""
    J_hot = dupuy_J(mu_h, p["mu_cold"], p["rh"], p["k_md"], p["h"], p.get("rc"))
    return torch.maximum(J_hot * p.get("J_mult", 1.0), dupuy_J_cold(p))


def dupuy_J_cold(p):
    """Unstimulated (cold) Dupuy productivity, m3/s per Pa."""
    k = p["k_md"] * 9.869233e-16
    rc = p.get("rc", ROCK["rc"])
    ln = torch.log(rc / ROCK["rw"]) if torch.is_tensor(rc) else math.log(rc / ROCK["rw"])
    return 2 * math.pi * k * p["h"] / (p["mu_cold"] * ln) * p.get("J_mult", 1.0)


# =============================================================================
# 2. RESERVOIR PINN
# =============================================================================
TAU_MAX = 400.0
ALPHA0 = ROCK["lam"] / (ROCK["rho_rock"] * ROCK["c_rock"])
# normalisation ranges -> [-1, 1]; also the collocation sampling domain
RES_RANGES = {"rh": (2.0, 70.0, "log"), "h": (10.0, 35.0, "lin"), "k_md": (50.0, 500.0, "log"),
              "mu_cold": (2.0, 20.0, "log"), "mu_hot": (0.05, 1.0, "log"),
              "T0": (295.0, 330.0, "lin"), "Ts": (480.0, 545.0, "lin"),
              "alpha": (0.7 * ALPHA0, 1.3 * ALPHA0, "log")}
RES_KEYS = list(RES_RANGES)


def _norm(v, lo, hi, kind):
    if kind == "log":
        v, lo, hi = torch.log(v), math.log(lo), math.log(hi)
    return torch.clamp(2 * (v - lo) / (hi - lo) - 1, -1.5, 1.5)


def _sample(n, lo, hi, kind, device):
    u = torch.rand(n, 1, device=device)
    if kind == "log":
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    return lo + u * (hi - lo)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, n_hidden, act=nn.Tanh):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReservoirPINN(nn.Module):
    """theta(tau; params) = (T-T0)/(Ts-T0) with theta(0)=1 exactly, and a
    bounded log-rate correction delta on top of zonal Dupuy."""

    def __init__(self, width=128, n_hidden=4):
        super().__init__()
        self.mlp = MLP(3 + len(RES_KEYS), 2, width, n_hidden)

    def forward(self, tau, p):
        """tau: (N,1) days since end of injection; p: dict of (N,1) tensors."""
        tn = tau / TAU_MAX
        s1 = torch.sqrt(torch.clamp(tn, min=0) + 1e-12)
        feats = [tn, s1, torch.log1p(torch.clamp(tau, min=0)) / math.log1p(TAU_MAX)]
        feats += [_norm(p[k], *RES_RANGES[k]) for k in RES_KEYS]
        o = self.mlp(torch.cat(feats, dim=1))
        theta = torch.exp(-s1 * nn.functional.softplus(o[:, :1]))      # IC hard-coded
        delta = 0.5 * torch.tanh(o[:, 1:2])
        T = p["T0"] + theta * (p["Ts"] - p["T0"])
        mu = andrade_mu(T, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"])
        J = dupuy_J_zonal(mu, p) * torch.exp(delta)                    # m3/s/Pa
        return {"theta": theta, "delta": delta, "T": T, "mu": mu,
                "J_bpd_per_MPa": J * 1e6 * M3S_TO_BPD,
                "Q_ref_bpd": J * DP_REF * M3S_TO_BPD}


def well_params_tensor(w, n, device, v_steam_m3=None):
    wp = WELL_REGISTRY[w] if isinstance(w, str) else w
    f = lambda v: torch.full((n, 1), float(v), device=device)
    p = {k: f(wp[k]) for k in ["h", "k_md", "mu_cold", "mu_hot", "T0", "Ts"]}
    p["alpha"] = f(ALPHA0)
    if v_steam_m3 is None:
        v_steam_m3 = wp["Qs"] * wp["t_inj"] * 86400.0
    v = v_steam_m3 if torch.is_tensor(v_steam_m3) else f(v_steam_m3)
    p["rh"] = heated_radius(v, p["h"], p["T0"], p["Ts"], t_inj_s=v / wp["Qs"])
    p["rc"] = f(wp.get("rc", ROCK["rc"]))            # drainage radius (field-calibrated for Baghewala)
    p["J_mult"] = f(wp.get("J_mult", 1.0))           # productivity multiplier (field-calibrated)
    return p


def qc_reservoir_data(df, max_step_K=15.0):
    """Data-quality checks on production history. Returns (clean_df, qc_report).
    * non-physical temperature cliffs (> max_step_K drop in one sample): rows from the
      cliff onward are masked (seen in the ORIGINAL generator output, fixed in v2)
    * heated radius vs steam energy balance; cum oil vs integral of rate"""
    keep, rep_ = [], {}
    for w, g in df.groupby("well_id", sort=False):
        g = g.sort_values("t_days_since_production_start")
        r = {"rows": int(len(g))}
        steps = g["reservoir_temp_K"].diff()
        bad = steps < -max_step_K
        if bad.any():
            t_cut = float(g.loc[bad.idxmax(), "t_days_since_production_start"])
            r["temperature_cliff_at_day"] = t_cut
            r["rows_masked"] = int((g["t_days_since_production_start"] >= t_cut).sum())
            g = g[g["t_days_since_production_start"] < t_cut]
        dt = g["t_days_since_production_start"].diff().median()
        r["cum_oil_consistency_pct"] = round(float(
            100 * ((g["oil_rate_bbl_day"] * dt).cumsum().iloc[-1] / g["cum_oil_bbl"].iloc[-1] - 1)), 3)
        if w in WELL_REGISTRY:
            wp = WELL_REGISTRY[w]
            rh = float(heated_radius(torch.tensor(wp["Qs"] * wp["t_inj"] * 86400.0), wp["h"], wp["T0"], wp["Ts"]))
            r["heated_radius_vs_energy_balance_pct"] = round(100 * (g["heated_radius_m"].iloc[0] / rh - 1), 3)
        rep_[w] = r
        keep.append(g)
    return pd.concat(keep, ignore_index=True), rep_


def load_reservoir_data(df, device):
    """ALL wells, ALL columns: t (+ soak -> tau), T, heated radius (model input),
    oil rate (data loss). Cum oil is checked in validation. Well properties that the
    CSV does not carry (h, k, viscosities, T0, Ts) come from WELL_REGISTRY."""
    rows = {"tau": [], "theta": [], "logQ": [], "rate_mask": [], "p": {k: [] for k in RES_KEYS}}
    for w, g in df.groupby("well_id"):
        if w not in WELL_REGISTRY:
            print(f"  [WARN] {w} not in WELL_REGISTRY -- add its properties to use it")
            continue
        wp = WELL_REGISTRY[w]
        n = len(g)
        tau = torch.tensor(g["t_days_since_production_start"].values + wp["t_soak"],
                           dtype=torch.float32, device=device).unsqueeze(1)
        p = well_params_tensor(w, n, device)
        p["rh"] = torch.tensor(g["heated_radius_m"].values, dtype=torch.float32, device=device).unsqueeze(1)
        th = (torch.tensor(g["reservoir_temp_K"].values, dtype=torch.float32, device=device)
              .unsqueeze(1) - wp["T0"]) / (wp["Ts"] - wp["T0"])
        q = torch.tensor(np.log(np.maximum(g["oil_rate_bbl_day"].values, 1e-6)),
                         dtype=torch.float32, device=device).unsqueeze(1)
        rows["tau"].append(tau); rows["theta"].append(th); rows["logQ"].append(q)
        rows["rate_mask"].append(torch.ones((n, 1), device=device))
        for k in RES_KEYS:
            rows["p"][k].append(p[k])
    out = {k: torch.cat(v) for k, v in rows.items() if k != "p"}
    out["p"] = {k: torch.cat(v) for k, v in rows["p"].items()}
    return out


def sample_reservoir_collocation(n, device):
    p = {k: _sample(n, lo, hi, kind, device) for k, (lo, hi, kind) in RES_RANGES.items()}
    tau = TAU_MAX * torch.rand(n, 1, device=device) ** 2        # denser at early time
    return tau, p


def train_reservoir(model, data, iters, device, lr=2e-3, batch=512, n_coll=1024,
                    log_every=500, tag="Reservoir PINN", physics_weight=1.0, history=None):
    """physics_weight=0 trains the same network on data only (ablation)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.02)
    N = data["tau"].shape[0]
    t0 = time.time()
    for it in range(iters + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        pd_ = {k: v[idx] for k, v in data["p"].items()}
        out_d = model(data["tau"][idx], pd_)
        L_T = torch.mean((out_d["theta"] - data["theta"][idx]) ** 2)
        m = data["rate_mask"][idx]
        L_Q = torch.sum(m * (torch.log(out_d["Q_ref_bpd"]) - data["logQ"][idx]) ** 2) / m.sum().clamp(min=1)

        tau_c, pc = sample_reservoir_collocation(n_coll, device)
        tau_c.requires_grad_(True)
        out_c = model(tau_c, pc)
        with torch.no_grad():
            th_bl = theta_boberg_lantz(tau_c, pc["rh"], pc["h"], pc["alpha"])
        L_BL = torch.mean((out_c["theta"] - th_bl) ** 2)
        dth = torch.autograd.grad(out_c["theta"].sum(), tau_c, create_graph=True)[0] * TAU_MAX
        L_mono = torch.mean(torch.relu(dth) ** 2)                    # no heat source after injection
        L_delta = torch.mean(out_c["delta"] ** 2)                    # Dupuy prior

        loss = 10 * L_T + L_Q + physics_weight * (10 * L_BL + L_mono + 0.01 * L_delta)
        if history is not None and it % 25 == 0:
            history.append({"it": it, "data": float(10 * L_T + L_Q), "physics": float(10 * L_BL + L_mono)})
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % log_every == 0:
            print(f"  [{tag}] it {it:5d} | loss {loss.item():.3e} | T {L_T.item():.2e} "
                  f"| logQ {L_Q.item():.2e} | BL {L_BL.item():.2e} | mono {L_mono.item():.1e} "
                  f"| {time.time() - t0:.0f}s")
    return model


def fine_tune_reservoir(model, field_df, device, iters=800, lr=3e-4):
    """Assimilate real production history (same columns as reservoir_css_timeseries.csv,
    wells must be in WELL_REGISTRY). Physics terms stay active as regularisers."""
    data = load_reservoir_data(qc_reservoir_data(field_df)[0], device)
    return train_reservoir(model, data, iters, device, lr=lr, tag="Fine-tune")


# =============================================================================
# 3. SRP PINN  (Everitt & Jennings damped wave eq., diagnostic form)
# =============================================================================
N_HARM_NET = 6      # periodic time embedding in the network (sharp pump-card transitions)
N_HARM_BC = 8       # Fourier fit of the measured surface position (Gibbs-style)
U_SCALE = 0.05      # u = u_s(tau) + xi * U_SCALE * N(xi, tau)


def fourier_fit(tau, y, K=None):
    """Least-squares Fourier series of a periodic signal sampled at tau in [0,1)."""
    K = K or N_HARM_BC
    k = np.arange(1, K + 1)
    X = np.hstack([np.ones((len(tau), 1)), np.cos(2 * np.pi * np.outer(tau, k)),
                   np.sin(2 * np.pi * np.outer(tau, k))])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def damping_from_viscosity(mu_Pa_s, rod_diam_in, tubing_id_in, rod_rho_lbm_ft3=490.0):
    """Everitt-Jennings viscous damping (1/s) from annular Couette drag on the rod:
       drag per unit length = 2 pi mu v / ln(ID/OD);  c = drag coeff / rod mass per length.
    This ties SRP damping to the oil viscosity predicted by the reservoir PINN."""
    A = math.pi / 4 * (rod_diam_in * 0.0254) ** 2
    m_per_len = rod_rho_lbm_ft3 * 16.0185 * A
    return 2 * math.pi * mu_Pa_s / math.log(tubing_id_in / rod_diam_in) / m_per_len


def case_constants(c):
    A = math.pi / 4 * c["diam"] ** 2
    v = math.sqrt(144 * c["E"] * G_C / c["rho"])
    T = 60.0 / c["spm"]
    S = c["stroke"] / 12.0
    return dict(A=A, v=v, T=T, S=S, kappa=(c["L"] / (v * T)) ** 2,
                load_scale=c["E"] * A * S / c["L"],
                static=c["rho"] * A / 144 * c["L"] * (1 - 62.4 * c["sg"] / c["rho"]))


class SRPPINN(nn.Module):
    def __init__(self, card_df, width=120, n_hidden=4, damping_per_s=None, cases=None, feat_stats=None):
        """card_df : cards with columns case_id, t_sec_within_stroke, surface_position_ft,
                     surface_load_lbf (pump columns only needed for validation)
        cases     : {case_id: rod config}; default SRP_CASES. A config may carry its own
                    'damping' (1/s), e.g. from damping_from_viscosity().
        damping_per_s: float -> fixed default damping; None -> one learned damping.
                    NOTE: surface position+load alone do NOT identify damping; learn it
                    only with a downhole reference.
        feat_stats: (mean, std) of log case features -- frozen from the base model when
                    new SCADA cases are added, so old cases keep their encoding."""
        super().__init__()
        cases = dict(SRP_CASES if cases is None else cases)
        self.cases = cases
        self.case_ids = list(cases)
        raw = torch.log(torch.tensor([[c["diam"], c["L"], c["E"], c["rho"], c["spm"], c["stroke"], c["sg"]]
                                      for c in cases.values()], dtype=torch.float32))
        if feat_stats is None:
            feat_stats = (raw.mean(0), raw.std(0) + 1e-6)
        self.register_buffer("feat_mean", feat_stats[0].clone())
        self.register_buffer("feat_std", feat_stats[1].clone())
        self.register_buffer("feat", (raw - self.feat_mean) / self.feat_std)
        cc = [case_constants(c) for c in cases.values()]
        for k in ["T", "S", "kappa", "load_scale", "static", "A", "v"]:
            self.register_buffer(k, torch.tensor([x[k] for x in cc], dtype=torch.float32))
        coefs, sig = [], []
        for cid in self.case_ids:
            g = card_df[card_df.case_id == cid]
            if len(g) < 2 * N_HARM_BC + 1:
                raise ValueError(f"case {cid}: need >= {2 * N_HARM_BC + 1} card samples")
            cst = case_constants(cases[cid])
            coef = fourier_fit(g["t_sec_within_stroke"].values / cst["T"],
                               g["surface_position_ft"].values / cst["S"])
            coefs.append(coef)
            k = np.arange(1, N_HARM_BC + 1)
            a, b = coef[1:N_HARM_BC + 1], coef[N_HARM_BC + 1:]
            sig.append(math.sqrt(0.5 * np.sum(((2 * np.pi * k) ** 2) ** 2 * (a ** 2 + b ** 2))))
        self.register_buffer("bc_coef", torch.tensor(np.array(coefs), dtype=torch.float32))
        self.register_buffer("acc_rms", torch.tensor(sig, dtype=torch.float32))
        self.mlp = MLP(1 + 2 * N_HARM_NET + 7, 1, width, n_hidden)
        default_c = 1.0 if damping_per_s is None else damping_per_s
        self.register_buffer("log_c_case", torch.log(torch.tensor(
            [c.get("damping", default_c) for c in cases.values()], dtype=torch.float32)))
        self.learn_damping = damping_per_s is None
        if self.learn_damping:
            self.log_c = nn.Parameter(torch.tensor(math.log(0.3)))

    def damping(self, cidx):
        if self.learn_damping:
            return torch.exp(self.log_c).expand(cidx.shape[0]).unsqueeze(1)
        return torch.exp(self.log_c_case[cidx]).unsqueeze(1)

    def surface_motion(self, tau, cidx):
        k = torch.arange(1, N_HARM_BC + 1, device=tau.device, dtype=tau.dtype)
        w = 2 * math.pi * k
        c = self.bc_coef[cidx]
        a0, a, b = c[:, :1], c[:, 1:N_HARM_BC + 1], c[:, N_HARM_BC + 1:]
        cs, sn = torch.cos(w * tau), torch.sin(w * tau)
        u = a0 + (a * cs + b * sn).sum(1, keepdim=True)
        du = (w * (-a * sn + b * cs)).sum(1, keepdim=True)
        d2u = (-(w ** 2) * (a * cs + b * sn)).sum(1, keepdim=True)
        return u, du, d2u

    def net(self, xi, tau, cidx):
        k = torch.arange(1, N_HARM_NET + 1, device=tau.device, dtype=tau.dtype)
        emb = torch.cat([torch.sin(2 * math.pi * k * tau), torch.cos(2 * math.pi * k * tau)], 1)
        return self.mlp(torch.cat([xi, emb, self.feat[cidx]], 1))

    def fields(self, xi, tau, cidx, need_pde=False):
        """Returns normalised displacement u, strain u_xi and (optionally) PDE residual."""
        if need_pde or not xi.requires_grad:
            xi = xi if xi.requires_grad else xi.clone().requires_grad_(True)
            tau = tau if tau.requires_grad else tau.clone().requires_grad_(True)
        N = self.net(xi, tau, cidx)
        us, dus, d2us = self.surface_motion(tau, cidx)
        u = us + xi * U_SCALE * N
        gN = torch.autograd.grad(N.sum(), [xi, tau], create_graph=True)
        N_x, N_t = gN
        u_x = U_SCALE * (N + xi * N_x)
        out = {"u": u, "u_x": u_x}
        if need_pde:
            N_xx = torch.autograd.grad(N_x.sum(), xi, create_graph=True)[0]
            N_tt = torch.autograd.grad(N_t.sum(), tau, create_graph=True)[0]
            u_xx = U_SCALE * (2 * N_x + xi * N_xx)
            u_t = dus + xi * U_SCALE * N_t
            u_tt = d2us + xi * U_SCALE * N_tt
            c = self.damping(cidx)
            kap, T = self.kappa[cidx].unsqueeze(1), self.T[cidx].unsqueeze(1)
            R = u_xx - kap * (u_tt + c * T * u_t)
            out["R"] = R / (kap * self.acc_rms[cidx].unsqueeze(1))
        return out

    @torch.no_grad()
    def predict_card(self, cid, n=200):
        """Surface + downhole (pump) card for a case, in field units."""
        ci = self.case_ids.index(cid)
        dev = self.feat.device
        with torch.enable_grad():
            tau = torch.linspace(0, 1, n, device=dev).unsqueeze(1)
            cidx = torch.full((n,), ci, device=dev, dtype=torch.long)
            s = self.fields(torch.zeros(n, 1, device=dev), tau, cidx)
            p = self.fields(torch.ones(n, 1, device=dev), tau, cidx)
        S, ls, st = self.S[ci].item(), self.load_scale[ci].item(), self.static[ci].item()
        return pd.DataFrame({"t_s": tau.squeeze().cpu().numpy() * self.T[ci].item(),
                             "surface_pos_ft": s["u"].squeeze().detach().cpu().numpy() * S,
                             "surface_load_lbf": s["u_x"].squeeze().detach().cpu().numpy() * ls + st,
                             "pump_pos_ft": p["u"].squeeze().detach().cpu().numpy() * S,
                             "pump_load_lbf": p["u_x"].squeeze().detach().cpu().numpy() * ls + st})


def load_srp_data(model, df, device):
    df = df[df.case_id.isin(model.case_ids)]
    cid = torch.tensor([model.case_ids.index(c) for c in df.case_id], device=device)
    T = model.T[cid.cpu()].to(device)
    tau = torch.tensor(df.t_sec_within_stroke.values, dtype=torch.float32, device=device) / T
    ls = model.load_scale[cid.cpu()].to(device)
    st = model.static[cid.cpu()].to(device)
    F = (torch.tensor(df.surface_load_lbf.values, dtype=torch.float32, device=device) - st) / ls
    return {"cidx": cid, "tau": tau.unsqueeze(1), "F_n": F.unsqueeze(1)}


def train_srp(model, data, iters, device, lr=2e-3, batch=512, n_coll=1024, log_every=500,
              coll_cases=None, tag="SRP PINN", pde_weight=1.0, history=None):
    """coll_cases: case indices for PDE collocation (default: all).
    pde_weight=0 trains the same network on surface data only (ablation)."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.02)
    N, nc = data["tau"].shape[0], len(model.case_ids)
    t0 = time.time()
    for it in range(iters + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        cidx = data["cidx"][idx]
        o = model.fields(torch.zeros(batch, 1, device=device), data["tau"][idx], cidx)
        L_load = torch.mean(((o["u_x"] - data["F_n"][idx]) / U_SCALE) ** 2)
        xi = torch.rand(n_coll, 1, device=device)
        tau = torch.rand(n_coll, 1, device=device)
        if coll_cases is None:
            cc = torch.randint(0, nc, (n_coll,), device=device)
        else:
            pool = torch.tensor(coll_cases, device=device)
            cc = pool[torch.randint(0, len(pool), (n_coll,), device=device)]
        L_pde = torch.mean(model.fields(xi, tau, cc, need_pde=True)["R"] ** 2)
        loss = L_load + pde_weight * L_pde
        if history is not None and it % 25 == 0:
            history.append({"it": it, "data": float(L_load), "physics": float(L_pde)})
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if model.learn_damping:
            with torch.no_grad():
                model.log_c.clamp_(math.log(0.01), math.log(20.0))
        if it % log_every == 0:
            print(f"  [{tag}] it {it:5d} | loss {loss.item():.3e} | load {L_load.item():.2e} "
                  f"| wave-PDE {L_pde.item():.2e} "
                  f"| {time.time() - t0:.0f}s")
    return model


# =============================================================================
# 4. CARD DIAGNOSTICS (rod floating, fluid pound / impact, fillage, unsetting)
# =============================================================================
def diagnose_card(surface_load, pump_pos, pump_load, static_rod_wt, expected_fluid_load=None,
                  t=None, rod_EA_lbf=None, wave_speed_ft_s=None):
    """Card diagnostics.  CONVENTION (same as the data): position is rod displacement
    measured DOWNWARD from the top of stroke (min position = top), loads include the
    buoyed rod weight.  Upstroke = position decreasing = fluid load carried.
    Returns rod floating, pump state, fillage, and impact at plunger/fluid contact."""
    sl, pp, pl = map(np.asarray, (surface_load, pump_pos, pump_load))
    res = {"min_surface_load_lbf": round(float(sl.min()), 1), "max_surface_load_lbf": round(float(sl.max()), 1),
           "rod_float_margin": round(float(sl.min() / static_rod_wt), 3)}
    res["rod_floating"] = bool(sl.min() < 0.1 * static_rod_wt)
    rng = float(pl.max() - pl.min())
    res["pump_load_range_lbf"] = round(rng, 1)
    if expected_fluid_load and rng < 0.1 * expected_fluid_load:
        res["pump_state"] = "NO FLUID-LOAD TRANSFER (gas lock / worn valves / free-end synthetic card)"
        res["fillage"] = None
        return res
    n = len(pp)
    i_top, i_bot = int(np.argmin(pp)), int(np.argmax(pp))
    down = np.arange(i_top, i_top + ((i_bot - i_top) % n) + 1) % n        # top -> bottom in time
    top, bot = pp[i_top], pp[i_bot]
    mid = 0.5 * (pl.max() + pl.min())
    below = np.where(pl[down] < mid)[0]
    j = int(down[below[0]]) if len(below) else i_bot
    res["fillage"] = round(float(np.clip((bot - pp[j]) / max(bot - top, 1e-9), 0, 1)), 3)
    res["pump_state"] = "FLUID POUND" if res["fillage"] < 0.85 else "full pump"
    # fluid load from load PLATEAUS (robust to transition overshoot): median load over the
    # middle half of the upstroke minus median load after valve opening on the downstroke
    up = np.arange(i_bot, i_bot + ((i_top - i_bot) % n) + 1) % n
    up_mid = up[len(up) // 4: max(len(up) * 3 // 4, len(up) // 4 + 1)]
    k = int(np.where(down == j)[0][0]) if j in down else len(down) - 1
    after = down[k:]
    after = after[len(after) // 4: max(len(after) * 3 // 4, len(after) // 4 + 1)] if len(after) > 3 else after
    res["fluid_load_plateau_lbf"] = round(float(np.median(pl[up_mid]) - np.median(pl[after])), 1)
    if t is not None and rod_EA_lbf and wave_speed_ft_s:
        tt = np.asarray(t)
        vel = np.gradient(pp, tt)
        v_imp = abs(float(vel[j])) if res["fillage"] < 0.9 else 0.0
        res["impact_velocity_ft_s"] = round(v_imp, 3)
        res["impact_load_lbf"] = round(rod_EA_lbf / wave_speed_ft_s * v_imp, 1)
    return res


def simulate_dynamometer_card(rod, fillage=1.0, damping_per_s=1.0, fluid_load_lbf=None,
                              plunger_in=1.5, n_nodes=40, n_strokes=4, n_out=400):
    """Forward Everitt-Jennings model WITH a pump boundary condition (independent of the
    PINN). Rod: dict(diam, L, E, rho, spm, stroke, sg). Pump: fluid load carried on the
    upstroke; on the downstroke the plunger falls through (1 - fillage) of its stroke in
    gas before hitting liquid (fluid pound), then the traveling valve opens and stays
    open (latched) until the upstroke. Stroke halves follow the polished-rod motion
    delayed by the wave travel time L/v.  Explicit FD, u positive downward, top
    Dirichlet (SHM), bottom Neumann EA u_x = F_pump. Returns the last stroke."""
    cst = case_constants(rod)
    L, EA, v = rod["L"], rod["E"] * cst["A"], cst["v"]
    W_f = fluid_load_lbf if fluid_load_lbf is not None else 0.340 * rod["sg"] * plunger_in ** 2 * L
    dx = L / n_nodes
    T = cst["T"]
    steps = int(math.ceil(T / (0.5 * dx / v)))
    dt = T / steps
    amp = cst["S"] / 2
    r2 = (v * dt / dx) ** 2
    c = damping_per_s
    U_prev = np.zeros(n_nodes + 1); U = np.zeros(n_nodes + 1)
    F, ramp, lag = 0.0, 0.03, L / v
    u_top, u_bot, Sp = 0.0, 2 * amp, 2 * amp
    phase_prev, released = None, False
    out = []
    for n in range(steps * n_strokes):
        t = (n + 1) * dt
        U_next = np.empty_like(U)
        U_next[1:-1] = (2 * U[1:-1] - U_prev[1:-1] * (1 - c * dt / 2)
                        + r2 * (U[2:] - 2 * U[1:-1] + U[:-2])) / (1 + c * dt / 2)
        U_next[0] = amp * (1 - math.cos(2 * math.pi * t / T))
        U_next[-1] = (2 * U[-1] - U_prev[-1] * (1 - c * dt / 2)
                      + r2 * (2 * U[-2] - 2 * U[-1] + 2 * dx * F / EA)) / (1 + c * dt / 2)
        up = U_next[-1]
        phase = "down" if ((t - lag) / T) % 1.0 < 0.5 else "up"
        if phase != phase_prev:
            if phase == "down":
                u_top, released = up, False
            else:
                u_bot = up
                Sp = max(u_bot - u_top, 1e-6)
            phase_prev = phase
        if phase == "up":             # traveling valve closed: fluid load picked up
            F = W_f * min(1.0, max(u_bot - up, 0.0) / (ramp * Sp))
        elif not released:            # plunger falls through gas until liquid contact
            u_contact = u_top + (1 - fillage) * Sp
            F = W_f * float(np.clip(1 - (up - u_contact) / (ramp * Sp), 0, 1))
            released = F <= 0.0
        else:
            F = 0.0
        U_prev, U = U, U_next
        if n >= steps * (n_strokes - 1):
            out.append((t - T * (n_strokes - 1), U[0], EA * (-3 * U[0] + 4 * U[1] - U[2]) / (2 * dx),
                        U[-1], F))
    o = np.array(out)
    o = o[np.linspace(0, len(o) - 1, n_out).astype(int)]
    return pd.DataFrame({"t_sec_within_stroke": o[:, 0], "surface_position_ft": o[:, 1],
                         "surface_load_lbf": o[:, 2] + cst["static"], "pump_position_ft": o[:, 3],
                         "pump_load_lbf": o[:, 4] + cst["static"]})


def card_work(pos, load):
    """Work per stroke (ft-lbf) from a closed card; u positive down -> W = -closed int F du."""
    p, f = np.r_[pos, pos[:1]], np.r_[load, load[:1]]
    return -float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(p)))


def diagnose_scada_card(base_model, base_cards, card_df, rod, damping_per_s=None, mu_fluid_Pa_s=None,
                        tubing_id_in=2.441, plunger_in=1.5, net_lift_ft=None, iters=800, device="cpu",
                        case_id="SCADA_CARD", pde_weight=1.0, cache=None):
    """Diagnose a NEW measured surface card (VFD/SRP data) with the SRP PINN.
    1. Adds the well's rod configuration as a new case (feature scaling frozen from the
       base model) and warm-starts from the trained network.
    2. Damping: given, or from fluid viscosity (damping_from_viscosity) -- e.g. the oil
       viscosity the reservoir PINN predicts for the current temperature.
    3. Short physics-constrained fine-tune on this card only (surface load + wave PDE).
    4. Predicts the downhole pump card and runs diagnose_card().
    5. Card-quality checks (tested in verify.py):
       * energy balance: polished-rod work > pump work > 0 (friction dissipates energy);
         catches load/position de-synchronisation and sensor faults;
       * fluid-load ratio: downhole fluid load (upstroke-downstroke load plateaus) /
         expected fluid load (0.340 SG Dp^2 H)
         within 0.70-1.35; catches load-cell calibration errors (or wrong fluid level/SG).
       A wrong pump depth / rod length shows up through the fluid-load ratio (the cause
       is then ambiguous with a load-cell error -- check both).
    card_df columns: t_sec_within_stroke, surface_position_ft, surface_load_lbf (one stroke)."""
    if damping_per_s is None:
        damping_per_s = (damping_from_viscosity(mu_fluid_Pa_s, rod["diam"], tubing_id_in, rod["rho"])
                         if mu_fluid_Pa_s is not None else SRP_DAMPING_PER_S)
    rod = dict(rod, damping=damping_per_s)
    cases = dict(base_model.cases); cases[case_id] = rod
    card = card_df.copy(); card["case_id"] = case_id
    allc = pd.concat([base_cards[["case_id", "t_sec_within_stroke", "surface_position_ft",
                                  "surface_load_lbf"]], card[["case_id", "t_sec_within_stroke",
                                                              "surface_position_ft", "surface_load_lbf"]]])
    m = SRPPINN(allc, damping_per_s=SRP_DAMPING_PER_S, cases=cases,
                feat_stats=(base_model.feat_mean.cpu(), base_model.feat_std.cpu())).to(device)
    # warm start: continue from the previous card's solution (real-time use); else from the base model
    m.mlp.load_state_dict(cache["mlp"] if cache and "mlp" in cache else base_model.mlp.state_dict())
    ci = m.case_ids.index(case_id)
    data = load_srp_data(m, card, device)
    train_srp(m, data, iters, device, lr=1e-3, batch=min(512, len(card)), n_coll=768,
              log_every=max(iters, 1), coll_cases=[ci], tag="SCADA fine-tune", pde_weight=pde_weight)
    if cache is not None:
        cache["mlp"] = {k: v.detach().clone() for k, v in m.mlp.state_dict().items()}
    pred = m.predict_card(case_id, n=len(card))
    cst = case_constants(rod)
    fit_err = float(np.sqrt(np.mean((np.interp(card.t_sec_within_stroke, pred.t_s, pred.surface_load_lbf)
                                     - card.surface_load_lbf) ** 2)))
    load_rng = float(card.surface_load_lbf.max() - card.surface_load_lbf.min())
    W_f = 0.340 * rod["sg"] * plunger_in ** 2 * (net_lift_ft or rod["L"])
    diag = diagnose_card(pred.surface_load_lbf, pred.pump_pos_ft, pred.pump_load_lbf, cst["static"],
                         expected_fluid_load=W_f, t=pred.t_s.values, rod_EA_lbf=rod["E"] * cst["A"],
                         wave_speed_ft_s=cst["v"])
    # Rod floating is read from the MEASURED surface card (the PINN reconstruction smooths
    # load dips); fillage / impact come from the PINN downhole card.
    meas_min = float(card.surface_load_lbf.min())
    diag["min_surface_load_lbf"] = round(meas_min, 1)
    diag["max_surface_load_lbf"] = round(float(card.surface_load_lbf.max()), 1)
    diag["rod_float_margin"] = round(meas_min / cst["static"], 3)
    diag["rod_floating"] = bool(meas_min < 0.1 * cst["static"])
    W_s = card_work(pred.surface_pos_ft.values, pred.surface_load_lbf.values)
    W_p = card_work(pred.pump_pos_ft.values, pred.pump_load_lbf.values)
    ratio_e = W_p / W_s if W_s > 0 else float("nan")
    ratio_f = diag.get("fluid_load_plateau_lbf", diag["pump_load_range_lbf"]) / W_f
    issues = []
    if not (W_s > 0 and 0.0 < ratio_e < 1.0):
        issues.append("energy balance violated -> load/position out of sync or sensor fault")
    if ratio_f > 1.35:
        issues.append("downhole load too high for configuration -> load cell reads high, "
                      "pump depth / rod length set too short, or fluid heavier than configured")
    elif ratio_f < 0.70:
        issues.append("downhole load too low -> fluid level above pump, leaking valves, "
                      "gas interference or load cell reads low")
    diag.update({"damping_used_per_s": round(damping_per_s, 3),
                 "surface_work_ft_lbf": round(W_s, 0), "pump_work_ft_lbf": round(W_p, 0),
                 "pump_to_surface_work_ratio": round(ratio_e, 3), "fluid_load_ratio": round(ratio_f, 3),
                 "surface_fit_rms_pct_of_range": round(100 * fit_err / max(load_rng, 1e-6), 2),
                 "card_quality": "OK" if not issues else "SUSPECT", "card_issues": issues})
    return diag, pred


# =============================================================================
# 5. SRP OPERATION OPTIMIZER (fast analytical load model, vectorised grid)
# =============================================================================
def srp_operating_state(spm, stroke_in, T_res, J_bpd_per_MPa, q_water_bpd, wp, cfg, run_frac=1.0):
    """Vectorised over spm/stroke/run_frac arrays. run_frac = pump-off-controller
    run time fraction (idle time lets fluid build up so the pump fills)."""
    N, S = np.asarray(spm, float), np.asarray(stroke_in, float)
    rf = np.broadcast_to(np.asarray(run_frac, float), N.shape)
    Dp, L_ft = cfg["plunger_in"], cfg["pump_depth_m"] / 0.3048
    A_r = math.pi / 4 * cfg["rod_diam_in"] ** 2
    E = cfg["rod_E_psi"]
    v_wave = math.sqrt(144 * E * G_C / 490.0)
    pr, pmin = cfg["p_reservoir_Pa"], cfg["pw_min_Pa"]
    J = max(J_bpd_per_MPa, 1e-9) / 1e6                      # bpd per Pa
    q_max_oil = J * (pr - pmin)
    Sp = S.copy()
    for _ in range(2):                                        # plunger stroke <- rod stretch
        PD = 0.1166 * Dp ** 2 * Sp * N
        C = PD * (1 - cfg["slippage"]) * rf
        q_liq_max = q_max_oil + q_water_bpd
        pumped_off = C >= q_liq_max
        fill = np.where(pumped_off, q_liq_max / np.maximum(C, 1e-9), 1.0)
        q_oil = np.where(pumped_off, q_max_oil, np.clip(C - q_water_bpd, 0, None))
        q_liq = np.minimum(q_oil + q_water_bpd, np.maximum(C, 1e-9))
        pw = np.where(pumped_off, pmin, np.clip(pr - q_oil / J, pmin, pr))
        wc = q_water_bpd / np.maximum(q_oil + q_water_bpd, 1e-9)
        sg_l = (1 - wc) * cfg["oil_sg"] + wc * 1.0
        rho_l = 1000.0 * sg_l
        subm_ft = np.maximum(pw - cfg["p_casing_Pa"], 0) / (rho_l * 9.81) / 0.3048
        H_ft = np.clip(L_ft - subm_ft, 0, L_ft)
        W_f = 0.340 * sg_l * Dp ** 2 * H_ft
        Sp = np.maximum(S - W_f * L_ft * 12 / (E * A_r), 0.3 * S)
    W_r = cfg["rod_weight_lb_ft"] * L_ft
    W_rf = W_r * (1 - 62.4 * sg_l / 490.0)
    alpha = S * N ** 2 / 70471.0
    # viscous rod drag (annular Couette), tubing fluid viscosity from reservoir temperature
    T_tub = cfg["T_ambient_K"] + cfg["tubing_temp_fraction"] * (T_res - cfg["T_ambient_K"])
    T_tub = max(T_tub, wp["T0"])
    mu_o = float(andrade_mu(torch.tensor(T_tub), torch.tensor(wp["mu_cold"]),
                            torch.tensor(wp["mu_hot"]), torch.tensor(wp["T0"]), torch.tensor(wp["Ts"])))
    # LIMITATION: log-mixing of oil and water viscosity. Heavy-oil water-in-oil emulsions
    # (below ~60-70 % water cut) can be MORE viscous than the oil, so rod drag / floating
    # risk is underestimated at intermediate water cuts. Replace with a measured or
    # Richardson/Pal-Rhodes emulsion correlation when Baghewala fluid data are available.
    mu_l = mu_o ** (1 - wc) * cfg["water_visc_Pa_s"] ** wc
    v_max = math.pi * (S / 12) * N / 60 * 0.3048             # m/s
    c_d = 2 * math.pi * mu_l * cfg["pump_depth_m"] / math.log(cfg["tubing_id_in"] / cfg["rod_diam_in"])
    F_drag = c_d * v_max / 4.448                              # lbf
    PPRL = W_f + W_rf + W_r * alpha + F_drag
    MPRL = W_rf - W_r * alpha - F_drag
    RFI = (W_r * alpha + F_drag) / W_rf                       # >=1 : rods float (MPRL <= 0)
    Smax, Smin = PPRL / A_r, np.maximum(MPRL, 0) / A_r
    SA = (cfg["rod_tensile_psi"] / 4 + 0.5625 * Smin) * cfg["goodman_SF"]
    goodman = Smax / SA
    CBE = W_rf + 0.5 * W_f
    torque = np.maximum(PPRL - CBE, CBE - MPRL) * S / 2
    # impact at plunger/fluid contact on the downstroke (fluid pound)
    phi = np.arccos(np.clip(1 - 2 * (1 - fill), -1, 1))
    v_imp = (Sp / 12 / 2) * (2 * math.pi * N / 60) * np.sin(phi) * (fill < 0.999)
    dF_imp = (E * A_r / v_wave) * v_imp
    # energy
    q_liq_m3s = q_liq * 0.158987 / 86400
    P_h = rho_l * 9.81 * H_ft * 0.3048 * q_liq_m3s
    P_f = c_d * v_max ** 2 / 2
    P_motor = (P_h / cfg["eta_pump"] + P_f) / (cfg["eta_surface"] * cfg["eta_motor"])
    kwh_day = P_motor * 24 / 1000 * rf
    return dict(spm=N, stroke_in=S, vfd_Hz=60 * N / cfg["spm_at_60Hz"], q_oil=q_oil, q_liq=q_liq,
                run_frac=rf, pw_MPa=pw / 1e6, fillage=fill,
                vol_eff=q_liq / np.maximum(0.1166 * Dp ** 2 * S * N * rf, 1e-9),
                PPRL=PPRL, MPRL=MPRL, rod_float_index=RFI, goodman=goodman, torque=torque,
                impact_lbf=dF_imp, unset_index=(dF_imp + np.maximum(-MPRL, 0)) / cfg["holddown_lbf"],
                mu_liquid=mu_l, kwh_day=kwh_day, drag_lbf=F_drag)


def optimize_srp(T_res, J_bpd_per_MPa, q_water_bpd, wp, cfg):
    spm = np.arange(cfg["vfd_Hz_min"], cfg["vfd_Hz_max"] + 1e-9, 0.75) * cfg["spm_at_60Hz"] / 60
    Ng, Sg, Rg = np.meshgrid(spm, cfg["strokes_in"], np.arange(0.25, 1.0001, 0.05))
    st = srp_operating_state(Ng.ravel(), Sg.ravel(), T_res, J_bpd_per_MPa, q_water_bpd, wp, cfg,
                             run_frac=Rg.ravel())
    hard = ((st["rod_float_index"] > cfg["rod_float_limit"]) | (st["goodman"] > cfg["goodman_limit"]) |
            (st["PPRL"] > cfg["structure_lbf"]) | (st["torque"] > cfg["gearbox_inlb"]))
    soft = (60.0 * np.maximum(0, cfg["fillage_target"] - st["fillage"]) * 10
            + 50.0 * np.maximum(0, st["impact_lbf"] / cfg["impact_allow_lbf"] - 1))
    score = st["q_oil"] * cfg["oil_price"] - st["kwh_day"] * cfg["power_cost_kwh"] - soft
    viol = (np.maximum(0, st["rod_float_index"] - cfg["rod_float_limit"])
            + np.maximum(0, st["goodman"] - cfg["goodman_limit"]))
    score = np.where(hard, -1e9 - 1e6 * viol, score)
    i = int(np.argmax(score))
    res = {k: (float(v[i]) if np.ndim(v) else float(v)) for k, v in st.items()}
    res["feasible"] = bool(~hard[i])
    return res


def srp_feedback_step(diag, setpoint, cfg, float_margin_target=0.20, fill_target=0.90, probe=1.08,
                      model_setpoint=None):
    """Closed loop: measured card diagnosis -> corrected SRP set-point.
    diag: output of diagnose_scada_card (surface-card floating margin, PINN fillage,
    card quality). setpoint: {'spm', 'stroke_in', 'run_frac'}. The model-based optimiser
    proposes; the measurement corrects. Safety corrections always win.
      * SUSPECT card      -> hold set-point, raise sensor alert (never act on bad data)
      * rod floating      -> cut SPM so min polished-rod load >= target margin
                             (floating index taken as ~ proportional to rod speed)
      * fluid pound       -> match displacement to inflow: SPM x fillage/target; at the VFD
                             minimum reduce pump-off run time instead
      * full & safe       -> probe SPM up (possible under-pumping), capped by the model optimum
    Returns (new_setpoint, action)."""
    spm_min = cfg["vfd_Hz_min"] * cfg["spm_at_60Hz"] / 60
    spm_max = cfg["vfd_Hz_max"] * cfg["spm_at_60Hz"] / 60
    sp = dict(setpoint)
    issues = diag.get("card_issues", [])
    sensor_fault = any(i.startswith("energy balance violated") for i in issues)
    if sensor_fault:                       # measurement itself not trustworthy
        return sp, "HOLD + SENSOR ALERT: " + "; ".join(issues)
    # Safety first: floating comes from the MEASURED surface card, independent of the PINN.
    m = diag["rod_float_margin"]
    if m < float_margin_target:
        rfi, rfi_t = 1.0 - m, 1.0 - float_margin_target
        sp["spm"] = max(spm_min, sp["spm"] * min(1.0, rfi_t / max(rfi, 1e-6)))
        note = " | model mismatch flagged (likely viscosity/damping) -> recalibrate" if issues else ""
        return sp, f"CUT SPM: rod floating risk (min load margin {m:.2f} < {float_margin_target}){note}"
    if issues:                             # model/configuration mismatch: do not trust PINN fillage
        return sp, "HOLD + RECALIBRATE: " + "; ".join(issues)
    f = diag.get("fillage")
    if f is not None and f < cfg["fillage_target"]:
        new = sp["spm"] * f / fill_target
        if new >= spm_min:
            sp["spm"] = new
            return sp, f"CUT SPM: fluid pound (fillage {f:.2f})"
        sp["spm"] = spm_min
        sp["run_frac"] = max(0.25, sp.get("run_frac", 1.0) * (new / spm_min))
        return sp, f"MIN SPM + shorter run time: fluid pound (fillage {f:.2f})"
    if f is not None and f >= 0.97:
        cap = model_setpoint["spm"] if model_setpoint else spm_max
        if sp.get("run_frac", 1.0) < 1.0:
            sp["run_frac"] = min(1.0, sp["run_frac"] * probe)
            return sp, "RAISE run time: pump full (possible under-pumping)"
        if sp["spm"] < min(cap, spm_max):
            sp["spm"] = min(sp["spm"] * probe, cap, spm_max)
            return sp, "RAISE SPM: pump full (possible under-pumping)"
    return sp, "HOLD: within targets"


# =============================================================================
# 6. CSS CYCLE OPTIMIZER (gradients through the reservoir PINN)
# =============================================================================
CSS_BOUNDS = {"t_inj": (3.0, FIELD_CONFIG["max_injection_days"]), "t_soak": (1.0, 10.0), "t_prod": (20.0, 300.0)}


def water_rate(t_prod_days, steam_cwe_bbl, cfg):
    return (cfg["condensate_return_frac"] * steam_cwe_bbl / cfg["condensate_decay_days"]
            * torch.exp(-t_prod_days / cfg["condensate_decay_days"]))


def cycle_economics(model_or_oracle, well, t_inj, t_soak, t_prod, cfg, n_grid=160):
    """Differentiable cycle KPIs for a batch of candidate schedules (each (B,1))."""
    wp = WELL_REGISTRY[well] if isinstance(well, str) else well
    dev = t_inj.device
    B = t_inj.shape[0]
    s = torch.linspace(0, 1, n_grid, device=dev).unsqueeze(0)          # (1,G)
    t_p = s * t_prod                                                    # (B,G)
    tau = (t_soak + t_p).reshape(-1, 1)
    v_steam = (wp["Qs"] * t_inj * 86400.0).repeat(1, n_grid).reshape(-1, 1)
    p = well_params_tensor(well, B * n_grid, dev, v_steam)
    dp_op = cfg["p_reservoir_Pa"] - cfg["pw_min_Pa"]
    if model_or_oracle == "oracle":
        th = theta_boberg_lantz(tau, p["rh"], p["h"], p["alpha"])
        T = p["T0"] + th * (p["Ts"] - p["T0"])
        J = dupuy_J_zonal(andrade_mu(T, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"]), p)
        q = J * dp_op * M3S_TO_BPD
    else:
        o = model_or_oracle(tau, p)
        q = o["J_bpd_per_MPa"] * dp_op / 1e6
    q = q.reshape(B, n_grid)
    q_cold = (dupuy_J_cold(p) * dp_op * M3S_TO_BPD).reshape(B, n_grid)[:, :1]   # no-steam alternative
    dt = t_prod / (n_grid - 1)
    oil = (dt * (q[:, :-1] + q[:, 1:]) / 2).sum(1, keepdim=True)
    steam_cwe = wp["Qs"] * t_inj * 86400.0 * ROCK["rho_steam"] / 1000.0 * 6.28981
    water = cfg["condensate_return_frac"] * steam_cwe * (1 - torch.exp(-t_prod / cfg["condensate_decay_days"]))
    kwh_per_bbl_liq = (1000 * 9.81 * cfg["pump_depth_m"] * 0.158987
                       / (cfg["eta_pump"] * cfg["eta_surface"] * cfg["eta_motor"]) / 3.6e6)
    kwh = (oil + water) * kwh_per_bbl_liq
    days = t_inj + t_soak + t_prod
    revenue = oil * cfg["oil_price"]
    cost = (steam_cwe * cfg["steam_cost_per_bbl_cwe"] + kwh * cfg["power_cost_kwh"]
            + days * cfg["opex_per_day"])
    sor = steam_cwe / oil.clamp(min=1e-3)
    # incremental economics vs. simply continuing cold production for the same days
    # (fixed well opex cancels; injection+soak days lose cold production)
    oil_cold = q_cold * days
    d_oil = oil - oil_cold
    d_kwh = d_oil.clamp(min=0) * kwh_per_bbl_liq + water * kwh_per_bbl_liq
    inc = d_oil * cfg["oil_price"] - steam_cwe * cfg["steam_cost_per_bbl_cwe"] - d_kwh * cfg["power_cost_kwh"]
    return {"oil_bbl": oil, "incremental_oil_vs_cold_bbl": d_oil,
            "incremental_profit_per_day": inc / days,
            "incremental_SOR": steam_cwe / d_oil.clamp(min=1e-3),
            "opex_coverage": revenue / (days * cfg["opex_per_day"]), "steam_cwe_bbl": steam_cwe, "SOR": sor, "kwh": kwh,
            "kwh_per_bbl_oil": kwh / oil.clamp(min=1e-3), "profit": revenue - cost,
            "profit_per_day": (revenue - cost) / days, "cost_per_bbl": cost / oil.clamp(min=1e-3),
            "days": days}


def optimize_css(model, well, cfg, device, n_starts=24, iters=400, lr=0.05):
    keys = list(CSS_BOUNDS)
    lo = torch.tensor([CSS_BOUNDS[k][0] for k in keys], device=device)
    hi = torch.tensor([CSS_BOUNDS[k][1] for k in keys], device=device)
    z = torch.randn(n_starts, 3, device=device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for _ in range(iters):
        x = lo + torch.sigmoid(z) * (hi - lo)
        k = cycle_economics(model, well, x[:, :1], x[:, 1:2], x[:, 2:3], cfg)
        pen = 200.0 * torch.relu(k["SOR"] - cfg["sor_limit"])        # SOR cap
        loss = -(k["incremental_profit_per_day"] - pen).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p_ in model.parameters():
        p_.requires_grad_(True)
    with torch.no_grad():
        x = lo + torch.sigmoid(z) * (hi - lo)
        k = cycle_economics(model, well, x[:, :1], x[:, 1:2], x[:, 2:3], cfg)
        score = k["incremental_profit_per_day"] - 200.0 * torch.relu(k["SOR"] - cfg["sor_limit"])
        i = int(torch.argmax(score))
    return {kk: float(x[i, j]) for j, kk in enumerate(keys)}


def oracle_grid_search(well, cfg, device):
    """Brute-force check of the CSS optimum on the analytical physics."""
    ti = torch.linspace(*CSS_BOUNDS["t_inj"], 23, device=device)
    ts = torch.linspace(*CSS_BOUNDS["t_soak"], 10, device=device)
    tp = torch.linspace(*CSS_BOUNDS["t_prod"], 29, device=device)
    G = torch.cartesian_prod(ti, ts, tp)
    ppd, sor = [], []
    for chunk in torch.split(G, 800):
        k = cycle_economics("oracle", well, chunk[:, :1], chunk[:, 1:2], chunk[:, 2:3], cfg, n_grid=80)
        ppd.append(k["incremental_profit_per_day"].squeeze(1)); sor.append(k["SOR"].squeeze(1))
    ppd, sor = torch.cat(ppd), torch.cat(sor)
    sc = ppd - 200.0 * torch.relu(sor - cfg["sor_limit"])
    j = int(torch.argmax(sc))
    frontier = []
    for cap in np.round(np.linspace(float(sor.min()) * 1.02, float(sor[j]), 5), 3):
        m = sor <= cap
        i = int(torch.argmax(torch.where(m, ppd, torch.tensor(-1e18))))
        frontier.append({"SOR_cap": float(cap), "incremental_profit_per_day": round(float(ppd[i]), 1),
                         "SOR": round(float(sor[i]), 3),
                         **{k: round(float(v), 2) for k, v in zip(CSS_BOUNDS, G[i].tolist())}})
    return dict(zip(CSS_BOUNDS, G[j].tolist())), float(sc[j]), frontier


DEFAULT_CYCLE_DECLINE = 0.85   # CAL=ASSUMED per-cycle OSR decline


@torch.no_grad()
def baghewala_field_forecast(model, schedule, cfg, device, n_wells=400, days=300, seed=0):
    """P10/P50/P90 forecast for wells drawn from the Baghewala condition ranges,
    at field drawdown, for a given CSS schedule."""
    wells = sample_baghewala_wells(n_wells, seed)
    t = np.arange(0, days + 1, 1.0)
    G = len(t)
    stack = lambda k: torch.tensor([w[k] for w in wells], dtype=torch.float32,
                                   device=device).repeat_interleave(G).unsqueeze(1)
    p = {k: stack(k) for k in ["h", "k_md", "mu_cold", "mu_hot", "T0", "Ts"]}
    p["alpha"] = torch.full_like(p["h"], ALPHA0)
    v = wells[0]["Qs"] * schedule["t_inj"] * 86400.0
    p["rh"] = heated_radius(torch.full_like(p["h"], v), p["h"], p["T0"], p["Ts"],
                            t_inj_s=schedule["t_inj"] * 86400.0)
    p["rc"] = stack("rc"); p["J_mult"] = stack("J_mult")
    tau = torch.tensor(np.tile(t + schedule["t_soak"], n_wells), dtype=torch.float32, device=device).unsqueeze(1)
    o = model(tau, p)
    dp = cfg["p_reservoir_Pa"] - cfg["pw_min_Pa"]
    T = o["T"].reshape(n_wells, G).cpu().numpy()
    q = (o["J_bpd_per_MPa"] * dp / 1e6).reshape(n_wells, G).cpu().numpy()
    # physics check for the same ensemble
    th = theta_boberg_lantz(tau, p["rh"], p["h"], p["alpha"])
    Tb = p["T0"] + th * (p["Ts"] - p["T0"])
    qb = (dupuy_J_zonal(andrade_mu(Tb, p["mu_cold"], p["mu_hot"], p["T0"], p["Ts"]), p) * dp * M3S_TO_BPD)
    qb = qb.reshape(n_wells, G).cpu().numpy()
    tp = min(int(schedule["t_prod"]), days)
    cum = _trapz(q[:, :tp + 1], t[:tp + 1], axis=1)
    q_cold = (dupuy_J_cold(p) * dp * M3S_TO_BPD).reshape(n_wells, G).cpu().numpy()[:, 0]
    d_oil = cum - q_cold * (schedule["t_inj"] + schedule["t_soak"] + tp)
    cum_b = _trapz(qb[:, :tp + 1], t[:tp + 1], axis=1)
    steam = wells[0]["Qs"] * schedule["t_inj"] * 86400.0 * ROCK["rho_steam"] / 1000.0 * 6.28981
    pc = lambda a: {f"P{k}": round(float(np.percentile(a, k)), 2) for k in (10, 50, 90)}
    return {
        "n_wells": n_wells, "schedule": schedule,
        "cycle_oil_bbl": pc(cum), "cycle_SOR": pc(steam / cum),
        "cold_rate_no_steam_bpd": pc(q_cold), "incremental_oil_vs_cold_bbl": pc(d_oil),
        "share_of_wells_where_steaming_adds_oil": round(float((d_oil > 0).mean()), 3),
        "peak_rate_bpd": pc(q.max(1)), "rate_day90_bpd": pc(q[:, 90]),
        "T_day90_K": pc(T[:, 90]), "T_day180_K": pc(T[:, 180]),
        "pinn_vs_physics_cycle_oil_MAPE_pct": round(float(np.mean(np.abs(cum / cum_b - 1)) * 100), 2),
        "_curves": {"t": t, "T": np.percentile(T, [10, 50, 90], axis=0),
                    "q": np.percentile(q, [10, 50, 90], axis=0)}}


def fit_cycle_decline(real_df):
    """Per-cycle incremental oil/steam ratio decline from REAL CSS records:
       ln(OSR_n) = a_well + b (n-1)  ->  decline factor d = exp(b)."""
    rows = []
    for w, g in real_df.sort_values("cycle_no").groupby("well_id"):
        s = g["cumulative_steam_injection_bbl"].diff().fillna(g["cumulative_steam_injection_bbl"])
        o = g["cumulative_oil_production_bbl"].diff().fillna(g["cumulative_oil_production_bbl"])
        for n, si, oi in zip(g["cycle_no"], s, o):
            if si > 0 and oi > 0:
                rows.append((w, n, math.log(oi / si)))
    d = pd.DataFrame(rows, columns=["well", "n", "lnosr"])
    d["x"] = d["n"] - 1
    xm = d["x"] - d.groupby("well")["x"].transform("mean")
    ym = d["lnosr"] - d.groupby("well")["lnosr"].transform("mean")
    b = float((xm * ym).sum() / (xm ** 2).sum())
    r2 = float(1 - ((ym - b * xm) ** 2).sum() / (ym ** 2).sum())
    out = {"fitted_decline_factor_per_cycle": round(math.exp(b), 4), "within_well_R2": round(r2, 4),
           "n_cycles_used": int(len(d)), "n_wells": int(d.well.nunique())}
    if r2 < 0.2 or math.exp(b) >= 1.0:
        out["decline_used"] = DEFAULT_CYCLE_DECLINE
        out["note"] = ("No identifiable cycle-to-cycle OSR decline in the real records (cycle-level "
                       "noise dominates); planning uses CAL=ASSUMED default. Refit on Baghewala "
                       "CSS cycle records.")
    else:
        out["decline_used"] = math.exp(b)
    return out


# =============================================================================
# 6b. FIELD STEAM ALLOCATION + RECOVERY FACTOR
# =============================================================================
RECOVERY_ASSUMPTIONS = {"oil_saturation": 0.80, "Bo": 1.05, "horizon_years": 10}   # CAL=ASSUMED


def _annualise(k):
    f = 365.0 / k["days"]
    return {"oil_yr": k["oil_bbl"] * f, "steam_yr": k["steam_cwe_bbl"] * f,
            "profit_yr": k["incremental_profit_per_day"] * 365.0, "cycles_yr": f}


def ooip_bbl(wp):
    """Oil in place in the drained cylinder (radius = calibrated drainage radius)."""
    rc = wp.get("rc", ROCK["rc"])
    v = math.pi * rc ** 2 * wp["h"] * ROCK["phi"] * RECOVERY_ASSUMPTIONS["oil_saturation"]
    return v * 6.28981 / RECOVERY_ASSUMPTIONS["Bo"]


def recovery_factor(oil_per_cycle, cycles_per_year, wp, decline):
    """Cumulative recovery over the horizon with per-cycle decline (real-field calibrated or assumed)."""
    n = RECOVERY_ASSUMPTIONS["horizon_years"] * cycles_per_year
    full = int(n)
    cum = sum(oil_per_cycle * decline ** i for i in range(full)) + (n - full) * oil_per_cycle * decline ** full
    return cum / ooip_bbl(wp)


@torch.no_grad()
def field_steam_allocation(model, cfg, device, n_wells=33, seed=7, decline=DEFAULT_CYCLE_DECLINE,
                           inj_options=(0, 4, 6, 8, 10, 12, 15, 18, 21, 25), t_soak=1.0, t_prod=300.0):
    """Allocate a FIXED annual field steam budget (what current practice uses) across wells.
    Each well picks one option: cold production (0 injection days) or a CSS schedule with
    t_inj in inj_options. Lagrangian relaxation: each well maximises profit - lam*steam,
    lam found by bisection so the field budget is met. All cycle KPIs from the PINN."""
    wells = sample_baghewala_wells(n_wells, seed)
    base_s = {"t_inj": 12.0, "t_soak": 2.0, "t_prod": 180.0}
    t = lambda v: torch.tensor([[float(v)]], dtype=torch.float32, device=device)
    rows = []
    for w in wells:
        kb = {k: float(v) for k, v in cycle_economics(model, w, t(12.0), t(2.0), t(180.0), cfg).items()}
        opts = []
        for ti in inj_options:
            if ti == 0:
                p = well_params_tensor(w, 1, device)
                q_cold = float(dupuy_J_cold(p)[0] * (cfg["p_reservoir_Pa"] - cfg["pw_min_Pa"]) * M3S_TO_BPD)
                opts.append({"t_inj": 0, "oil_yr": q_cold * 365, "steam_yr": 0.0, "profit_yr": 0.0,
                             "cycles_yr": 0.0, "oil_cycle": q_cold * 365})
            else:
                k = {kk: float(v) for kk, v in cycle_economics(model, w, t(ti), t(t_soak), t(t_prod), cfg).items()}
                a_ = _annualise(k)
                opts.append(dict(a_, t_inj=ti, oil_cycle=k["oil_bbl"]))
        rows.append({"well": w, "base": dict(_annualise(kb), oil_cycle=kb["oil_bbl"]), "opts": opts})
    budget = sum(r["base"]["steam_yr"] for r in rows)

    def pick(lam):
        ch = [max(r["opts"], key=lambda o: o["profit_yr"] - lam * o["steam_yr"]) for r in rows]
        return ch, sum(o["steam_yr"] for o in ch)
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, st = pick(mid)
        lo, hi = (mid, hi) if st > budget else (lo, mid)
    choice, steam_used = pick(hi)
    same = [next(o for o in r["opts"] if o["t_inj"] == 12) for r in rows]

    def tot(sel, sched_rows=None):
        oil = sum(o["oil_yr"] for o in sel); steam = sum(o["steam_yr"] for o in sel)
        prof = sum(o["profit_yr"] for o in sel)
        rf = float(np.mean([recovery_factor(o["oil_cycle"], o["cycles_yr"], r["well"], decline)
                            if o["cycles_yr"] > 0 else o["oil_yr"] * RECOVERY_ASSUMPTIONS["horizon_years"] / ooip_bbl(r["well"])
                            for o, r in zip(sel, rows)]))
        return {"field_oil_bpd": round(oil / 365, 1), "field_steam_bbl_cwe_per_yr": round(steam, 0),
                "field_SOR": round(steam / max(oil, 1e-9), 3), "incremental_profit_usd_per_yr": round(prof, 0),
                "mean_recovery_factor_10yr_pct": round(100 * rf, 1)}
    cur = tot([r["base"] for r in rows]); ss = tot(same); al = tot(choice)
    lam = hi
    hist = {}
    for o in choice:
        hist[o["t_inj"]] = hist.get(o["t_inj"], 0) + 1
    return {"n_wells": n_wells, "steam_budget_bbl_cwe_per_yr": round(budget, 0),
            "shadow_price_of_steam_usd_per_bbl_cwe": round(lam, 2),
            "current_practice_12_2_180": cur, "same_steam_per_cycle_12_1_300": ss,
            "allocated_same_field_budget": al,
            "allocation_injection_days_histogram": {str(k): v for k, v in sorted(hist.items())},
            "field_oil_change_pct_vs_current": round(100 * (al["field_oil_bpd"] / cur["field_oil_bpd"] - 1), 1),
            "field_SOR_change_pct_vs_current": round(100 * (al["field_SOR"] / cur["field_SOR"] - 1), 1),
            "recovery_assumptions": RECOVERY_ASSUMPTIONS,
            "per_well": [{"k_md": round(r["well"]["k_md"]), "h_m": round(r["well"]["h"], 1),
                          "mu_cold_Pa_s": round(r["well"]["mu_cold"], 1), "t_inj_days": o["t_inj"]}
                         for r, o in zip(rows, choice)]}


# =============================================================================
# 7. COUPLED DIGITAL-TWIN RUN (reservoir -> SRP, day by day)
# =============================================================================
@torch.no_grad()
def coupled_cycle_simulation(model, well, sched, cfg, device, use_optimizer=True):
    wp = WELL_REGISTRY[well]
    days = np.arange(0.0, sched["t_prod"], 1.0)
    tau = torch.tensor(days + sched["t_soak"], dtype=torch.float32, device=device).unsqueeze(1)
    p = well_params_tensor(well, len(days), device, wp["Qs"] * sched["t_inj"] * 86400.0)
    o = model(tau, p)
    steam_cwe = wp["Qs"] * sched["t_inj"] * 86400.0 * ROCK["rho_steam"] / 1000.0 * 6.28981
    qw = water_rate(torch.tensor(days), steam_cwe, cfg).numpy()
    rows = []
    for i, d in enumerate(days):
        T, J = float(o["T"][i]), float(o["J_bpd_per_MPa"][i])
        if use_optimizer:
            st = optimize_srp(T, J, qw[i], wp, cfg)
        else:
            st = srp_operating_state(np.array([cfg["baseline_spm"]]), np.array([cfg["baseline_stroke_in"]]),
                                     T, J, qw[i], wp, cfg)
            st = {k: float(np.ravel(v)[0]) for k, v in st.items()}
        rows.append({"day": d, "T_res_K": T, "rh_m": float(p["rh"][0]),
                     "oil_visc_res_Pa_s": float(o["mu"][i]), "water_bpd": float(qw[i]), **st})
    return pd.DataFrame(rows)


def summarise(df, label):
    oil = df.q_oil.sum()
    r = {"label": label, "cum_oil_bbl": round(oil, 1), "cum_water_bbl": round(df.water_bpd.sum(), 1),
            "energy_kwh": round(df.kwh_day.sum(), 1),
            "kwh_per_bbl_oil": round(df.kwh_day.sum() / max(oil, 1e-6), 2),
            "mean_fillage": round(df.fillage.mean(), 3), "mean_vol_eff": round(df.vol_eff.mean(), 3),
            "mean_run_time_frac": round(df.run_frac.mean(), 3),
            "days_rod_floating": int((df.rod_float_index >= 1).sum()),
            "days_impact_over_limit": int((df.impact_lbf > FIELD_CONFIG["impact_allow_lbf"]).sum()),
            "days_unsetting_risk": int((df.unset_index > 1).sum()),
            "max_goodman_loading": round(df.goodman.max(), 3),
            "max_impact_lbf": round(df.impact_lbf.max(), 0)}
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=here)
    ap.add_argument("--out", default=os.path.join(here, "dt_output"))
    ap.add_argument("--res-iters", type=int, default=3000)
    ap.add_argument("--srp-iters", type=int, default=3000)
    ap.add_argument("--well", default="BAGHEWALA_P50")
    ap.add_argument("--quick", action="store_true", help="smoke test (~1-2 min)")
    ap.add_argument("--load", action="store_true", help="load saved models, skip training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-calibration", action="store_true",
                    help="skip calibration of BAGHEWALA_P50 to public field-level rate / uplift")
    ap.add_argument("--learn-damping", action="store_true",
                    help="treat SRP damping as unknown (needs downhole reference to be identifiable)")
    a = ap.parse_args()
    if a.quick:
        a.res_iters, a.srp_iters = 200, 200
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    cfg = FIELD_CONFIG
    T0 = time.time()

    res_df_raw = pd.read_csv(os.path.join(a.data_dir, "reservoir_css_timeseries.csv"))
    res_df, qc = qc_reservoir_data(res_df_raw)
    print("Data QC (reservoir_css_timeseries.csv):")
    for w, r in qc.items():
        print(f"  {w:18s} {r}")
    print(f"  -> training on {len(res_df)} of {len(res_df_raw)} rows, {res_df.well_id.nunique()} wells pooled")
    srp_df = pd.read_csv(os.path.join(a.data_dir, "srp_dynamometer_microscale.csv"))
    real_df = pd.read_csv(os.path.join(a.data_dir, "real_field_css_cycle_data.csv"))

    res_model = ReservoirPINN().to(dev)
    srp_model = SRPPINN(srp_df, damping_per_s=None if a.learn_damping else SRP_DAMPING_PER_S).to(dev)
    n_res, n_srp = count_params(res_model), count_params(srp_model)
    print(f"Device {dev} | Reservoir PINN {n_res:,} | SRP PINN {n_srp:,} | TOTAL {n_res + n_srp:,} params")
    assert n_res + n_srp <= 100_000

    ck = os.path.join(a.out, "digital_twin_models.pt")
    if a.load and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        res_model.load_state_dict(s["reservoir"]); srp_model.load_state_dict(s["srp"])
    else:
        print("\n== Phase 1: Reservoir PINN (heating / cooling / production) ==")
        train_reservoir(res_model, load_reservoir_data(res_df, dev), a.res_iters, dev)
        print("\n== Phase 2: SRP PINN (surface card -> downhole card) ==")
        train_srp(srp_model, load_srp_data(srp_model, srp_df, dev), a.srp_iters, dev)
        torch.save({"reservoir": res_model.state_dict(), "srp": srp_model.state_dict(),
                    "field_config": cfg}, ck)
    report = {"params": {"reservoir_pinn": n_res, "srp_pinn": n_srp, "total": n_res + n_srp},
              "data_qc": qc, "rows_used": int(len(res_df)), "rows_in_file": int(len(res_df_raw))}

    # ---- validation: reservoir -------------------------------------------------
    print("\n== Validation ==")
    with torch.no_grad():
        val = {}
        for w, g in res_df.groupby("well_id"):
            wp = WELL_REGISTRY[w]
            tau = torch.tensor(g.t_days_since_production_start.values + wp["t_soak"],
                               dtype=torch.float32, device=dev).unsqueeze(1)
            pw_ = well_params_tensor(w, len(g), dev)
            pw_["rh"] = torch.tensor(g.heated_radius_m.values, dtype=torch.float32, device=dev).unsqueeze(1)
            o = res_model(tau, pw_)
            Tp = o["T"].squeeze().cpu().numpy()
            Qp = o["Q_ref_bpd"].squeeze().cpu().numpy()
            dt = float(g.t_days_since_production_start.diff().median())
            v = {"T_MAE_K": round(float(np.abs(Tp - g.reservoir_temp_K.values).mean()), 2),
                 "Q_MAPE_pct": round(float(np.mean(np.abs(Qp / g.oil_rate_bbl_day.values - 1)) * 100), 2),
                 "cum_oil_error_pct": round(float(100 * ((Qp * dt).sum() / g.cum_oil_bbl.values[-1] - 1)), 2)}
            val[w] = v
            print(f"  {w:18s} {v}")
    report["reservoir_validation"] = val

    # ---- validation: SRP (pump card was never used in training) -----------------
    srp_val, cards = {}, {}
    for cid, c in SRP_CASES.items():
        g = srp_df[srp_df.case_id == cid]
        cst = case_constants(c)
        card = srp_model.predict_card(cid, n=len(g))
        pos_err = np.abs(np.interp(g.t_sec_within_stroke, card.t_s, card.pump_pos_ft) - g.pump_position_ft).mean()
        sl_err = np.abs(np.interp(g.t_sec_within_stroke, card.t_s, card.surface_load_lbf) - g.surface_load_lbf).mean()
        pl_err = np.abs(np.interp(g.t_sec_within_stroke, card.t_s, card.pump_load_lbf) - g.pump_load_lbf).mean()
        diag = diagnose_card(card.surface_load_lbf, card.pump_pos_ft, card.pump_load_lbf, cst["static"],
                             expected_fluid_load=0.340 * c["sg"] * 1.5 ** 2 * c["L"])
        srp_val[cid] = {"surface_load_MAE_lbf": round(sl_err, 1), "pump_pos_MAE_ft": round(pos_err, 4),
                        "pump_load_MAE_lbf": round(pl_err, 1), "diagnosis": diag}
        cards[cid] = (g, card)
        print(f"  {cid:20s} surf-load MAE {sl_err:7.1f} lbf | pump-pos MAE {pos_err:.4f} ft | "
              f"pump-load MAE {pl_err:6.1f} lbf | {diag['pump_state']}")
    report["srp_validation"] = srp_val
    _lc = srp_model.log_c if srp_model.learn_damping else srp_model.log_c_case[0]
    report["srp_damping_per_s"] = {"value": round(math.exp(float(_lc)), 4),
                                   "mode": "learned" if a.learn_damping else "fixed (E&J)",
                                   "generator_true_value": 1.0}

    # ---- real-field calibration ------------------------------------------------
    decl = fit_cycle_decline(real_df)
    report["real_field_cycle_decline"] = decl
    print(f"  Real-field OSR decline per cycle: {decl}")

    # ---- CSS optimisation -------------------------------------------------------
    if a.well == "BAGHEWALA_P50" and not a.no_calibration:
        print("\n== Calibration of Baghewala P50 to public field-level data ==")
        cal = calibrate_baghewala(cfg)
        report["field_calibration"] = cal
        print(f"  {cal['calibrated']} | before {cal['before']} -> after {cal['after']}")
    print(f"\n== Phase 3: CSS cycle optimisation for {a.well} ==")
    wp = WELL_REGISTRY[a.well]
    base = {"t_inj": wp["t_inj"], "t_soak": wp["t_soak"], "t_prod": 180.0}
    opt = optimize_css(res_model, a.well, cfg, dev)
    orc, _, frontier = oracle_grid_search(a.well, cfg, dev)

    def kpis(model, s):
        t = lambda v: torch.tensor([[v]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            k = cycle_economics(model, a.well, t(s["t_inj"]), t(s["t_soak"]), t(s["t_prod"]), cfg)
        return {kk: round(float(v), 3) for kk, v in k.items()}

    css = {"baseline_schedule": base, "baseline_kpis_pinn": kpis(res_model, base),
           "optimal_schedule_pinn": {k: round(v, 2) for k, v in opt.items()},
           "optimal_kpis_pinn": kpis(res_model, opt),
           "optimal_kpis_physics_check": kpis("oracle", opt),
           "physics_grid_optimum": {k: round(v, 2) for k, v in orc.items()},
           "physics_grid_optimum_kpis": kpis("oracle", orc),
           "SOR_vs_profit_tradeoff": frontier,
           "bounds_active": [k for k, v in opt.items()
                             if min(abs(v - CSS_BOUNDS[k][0]), abs(v - CSS_BOUNDS[k][1])) < 0.02 * (CSS_BOUNDS[k][1] - CSS_BOUNDS[k][0])]}
    # balanced recommendation: lowest SOR that keeps >= 95 % of the max profit/day
    pmax = max(f["incremental_profit_per_day"] for f in frontier)
    bal = min((f for f in frontier if f["incremental_profit_per_day"] >= pmax - 0.05 * abs(pmax)),
              key=lambda f: f["SOR"])
    bal = {k: bal[k] for k in CSS_BOUNDS}
    css["recommended_balanced_schedule"] = bal
    css["recommended_balanced_kpis_pinn"] = kpis(res_model, bal)
    css["optimal_steam_cwe_bbl"] = css["recommended_balanced_kpis_pinn"]["steam_cwe_bbl"]
    b, o_ = css["baseline_kpis_pinn"], css["recommended_balanced_kpis_pinn"]
    css["delta_vs_baseline"] = {
        "incremental_profit_per_day_pct": round(100 * (o_["incremental_profit_per_day"]
                                                       / b["incremental_profit_per_day"] - 1), 1),
        "SOR_change_pct": round(100 * (o_["SOR"] / b["SOR"] - 1), 1),
        "oil_per_cycle_day_pct": round(100 * ((o_["oil_bbl"] / o_["days"]) / (b["oil_bbl"] / b["days"]) - 1), 1)}
    # multi-cycle plan with real-field decline
    d = decl["decline_used"]
    plan, cum_o, cum_s = [], 0.0, 0.0
    for n in range(1, 16):
        oil_n = o_["oil_bbl"] * d ** (n - 1)
        d_oil_n = o_["incremental_oil_vs_cold_bbl"] - o_["oil_bbl"] * (1 - d ** (n - 1))
        prof = (o_["incremental_profit_per_day"] * o_["days"]
                - o_["oil_bbl"] * (1 - d ** (n - 1)) * cfg["oil_price"])
        if d_oil_n <= 0 or prof <= 0 or o_["steam_cwe_bbl"] / oil_n > cfg["sor_limit"]:
            break
        cum_o += oil_n; cum_s += o_["steam_cwe_bbl"]
        plan.append({"cycle": n, "oil_bbl": round(oil_n, 1), "incremental_profit": round(prof, 0),
                     "cSOR": round(cum_s / cum_o, 2)})
    css["multi_cycle_plan"] = plan
    # marginal value of steam at the recommended schedule (physics oracle, +1 injection day)
    tt = lambda v: torch.tensor([[float(v)]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        k0 = cycle_economics("oracle", a.well, tt(bal["t_inj"]), tt(bal["t_soak"]), tt(bal["t_prod"]), cfg)
        k1 = cycle_economics("oracle", a.well, tt(bal["t_inj"] + 1), tt(bal["t_soak"]), tt(bal["t_prod"]), cfg)
    d_prof = float(k1["incremental_profit_per_day"] * k1["days"] - k0["incremental_profit_per_day"] * k0["days"])
    d_steam = float(k1["steam_cwe_bbl"] - k0["steam_cwe_bbl"])
    css["marginal_value_of_steam_usd_per_bbl_cwe"] = round(d_prof / d_steam, 2)
    css["steam_cost_usd_per_bbl_cwe"] = cfg["steam_cost_per_bbl_cwe"]
    css["binding_constraint"] = ("steam availability (max_injection_days) -- allocate generator capacity "
                                 "to the wells with the highest marginal value of steam"
                                 if "t_inj" in css["bounds_active"] else "economics (interior optimum)")
    # Same-steam schedule: keep the current steam per cycle, optimise soak and production days.
    # When steam supply is binding this is the schedule the field can actually adopt.
    best = None
    with torch.no_grad():
        for ts_ in np.arange(CSS_BOUNDS["t_soak"][0], CSS_BOUNDS["t_soak"][1] + 1e-9, 1.0):
            for tp_ in np.arange(60.0, CSS_BOUNDS["t_prod"][1] + 1e-9, 10.0):
                kk = cycle_economics(res_model, a.well, tt(base["t_inj"]), tt(ts_), tt(tp_), cfg)
                v = float(kk["incremental_profit_per_day"])
                if best is None or v > best[0]:
                    best = (v, {"t_inj": base["t_inj"], "t_soak": float(ts_), "t_prod": float(tp_)})
    sn = best[1]
    css["recommended_same_steam_schedule"] = sn
    css["recommended_same_steam_kpis_pinn"] = kpis(res_model, sn)
    ks = css["recommended_same_steam_kpis_pinn"]
    css["same_steam_delta_vs_baseline"] = {
        "incremental_profit_per_day_pct": round(100 * (ks["incremental_profit_per_day"] / b["incremental_profit_per_day"] - 1), 1),
        "SOR_change_pct": round(100 * (ks["SOR"] / b["SOR"] - 1), 1),
        "steam_per_year_change_pct": round(100 * ((ks["steam_cwe_bbl"] / ks["days"]) / (b["steam_cwe_bbl"] / b["days"]) - 1), 1),
        "oil_per_day_change_pct": round(100 * ((ks["oil_bbl"] / ks["days"]) / (b["oil_bbl"] / b["days"]) - 1), 1)}
    rec = sn if "t_inj" in css["bounds_active"] else bal
    css["schedule_used_downstream"] = "same-steam" if rec is sn else "balanced"
    print(f"  same-steam schedule {sn}: {css['same_steam_delta_vs_baseline']}")
    css["steaming_beats_cold_production"] = bool(o_["incremental_profit_per_day"] > 0)
    css["cycle_profitable_after_all_costs"] = bool(o_["profit"] > 0)
    report["css_optimisation"] = css
    for k in ["baseline_kpis_pinn", "optimal_schedule_pinn", "optimal_kpis_pinn",
              "optimal_kpis_physics_check", "physics_grid_optimum", "delta_vs_baseline",
              "SOR_vs_profit_tradeoff", "bounds_active", "recommended_balanced_schedule",
              "recommended_balanced_kpis_pinn"]:
        print(f"  {k}: {css[k]}")
    print(f"  economic cycles before shutdown: {len(plan)} | steaming beats cold: "
          f"{css['steaming_beats_cold_production']} | profitable after all costs: {css['cycle_profitable_after_all_costs']}")

    # ---- Baghewala field forecast (ensemble over field condition ranges) --------
    print("\n== Baghewala field forecast (P10/P50/P90 over published condition ranges) ==")
    ff_base = baghewala_field_forecast(res_model, base, cfg, dev)
    ff_rec = baghewala_field_forecast(res_model, rec, cfg, dev, seed=0)
    field = {"condition_ranges": {k: {"low": v[0], "P50": v[1], "high": v[2]} for k, v in BAGHEWALA_RANGES.items()},
             "current_schedule": {k: v for k, v in ff_base.items() if k != "_curves"},
             "recommended_schedule": {k: v for k, v in ff_rec.items() if k != "_curves"}}
    report["baghewala_field_forecast"] = field
    for k in ["current_schedule", "recommended_schedule"]:
        print(f"  {k}: {field[k]}")

    # ---- field-wide steam allocation (same annual steam budget) -----------------
    if a.well == "BAGHEWALA_P50":
        fa = field_steam_allocation(res_model, cfg, dev, decline=decl["decline_used"])
        report["field_steam_allocation"] = fa
        print(f"  field allocation: oil {fa['current_practice_12_2_180']['field_oil_bpd']} -> "
              f"{fa['allocated_same_field_budget']['field_oil_bpd']} bpd ({fa['field_oil_change_pct_vs_current']:+}%), "
              f"SOR {fa['field_SOR_change_pct_vs_current']:+}%, RF10 "
              f"{fa['current_practice_12_2_180']['mean_recovery_factor_10yr_pct']} -> "
              f"{fa['allocated_same_field_budget']['mean_recovery_factor_10yr_pct']} %")

    # ---- coupled SRP operation --------------------------------------------------
    print("\n== Phase 4: coupled daily run (reservoir PINN -> SRP optimizer) ==")
    df_opt = coupled_cycle_simulation(res_model, a.well, rec, cfg, dev, True)
    df_base = coupled_cycle_simulation(res_model, a.well, base, cfg, dev, False)
    df_opt.to_csv(os.path.join(a.out, "srp_daily_optimised.csv"), index=False)
    df_base.to_csv(os.path.join(a.out, "srp_daily_baseline_fixed_speed.csv"), index=False)
    s_o, s_b = summarise(df_opt, "optimised"), summarise(df_base, "fixed 8 SPM / 86 in")
    advice = []
    if s_o["mean_run_time_frac"] <= 0.3:
        advice.append("Pump is oversized for this inflow even at minimum VFD speed and 25 % run time: "
                      "downsize plunger / shorten stroke at next workover.")
    report["srp_operation"] = {"optimised (recommended CSS + SRP optimiser)": s_o,
                               "baseline (current CSS + fixed speed, no POC)": s_b, "advice": advice,
                               "setpoint_samples": df_opt.iloc[::15][["day", "T_res_K", "mu_liquid", "spm",
                                                                      "vfd_Hz", "stroke_in", "run_frac", "fillage",
                                                                      "rod_float_index", "impact_lbf"]]
                               .round(3).to_dict("records")}
    print(f"  {s_o}\n  {s_b}")
    # cold-production stress test (before steaming / late cold cycle): rod floating regime
    wpw = WELL_REGISTRY[a.well]
    with torch.no_grad():
        o_cold = res_model(torch.tensor([[TAU_MAX]], device=dev), well_params_tensor(a.well, 1, dev))
    J_cold = float(dupuy_J_cold(well_params_tensor(a.well, 1, dev))[0]) * M3S_TO_BPD * 1e6   # bpd per MPa
    cold_b = srp_operating_state(np.array([cfg["baseline_spm"]]), np.array([cfg["baseline_stroke_in"]]),
                                 wpw["T0"], J_cold, 0.0, wpw, cfg)
    cold_o = optimize_srp(wpw["T0"], J_cold, 0.0, wpw, cfg)
    pick = ["spm", "vfd_Hz", "stroke_in", "run_frac", "rod_float_index", "MPRL", "drag_lbf",
            "fillage", "impact_lbf", "q_oil", "kwh_day"]
    report["cold_well_rod_floating_test"] = {
        "T_K": wpw["T0"], "baseline": {k: round(float(np.ravel(cold_b[k])[0]), 3) for k in pick},
        "optimised": {k: round(cold_o[k], 3) for k in pick}}
    print(f"  cold-well test: {report['cold_well_rod_floating_test']}")

    # ---- expected benefits (problem statement) ----------------------------------
    report["expected_benefits_quantified"] = {
        "schedule": css["schedule_used_downstream"],
        "oil_per_day_change_pct": (css["same_steam_delta_vs_baseline"]["oil_per_day_change_pct"] if rec is sn
                                   else css["delta_vs_baseline"]["oil_per_cycle_day_pct"]),
        "SOR_change_pct": (css["same_steam_delta_vs_baseline"]["SOR_change_pct"] if rec is sn
                           else css["delta_vs_baseline"]["SOR_change_pct"]),
        "lifting_energy_kwh_per_bbl": {"baseline": s_b["kwh_per_bbl_oil"], "optimised": s_o["kwh_per_bbl_oil"]},
        "rod_floating_days": {"baseline": s_b["days_rod_floating"], "optimised": s_o["days_rod_floating"]},
        "impact_over_limit_days": {"baseline": s_b["days_impact_over_limit"],
                                   "optimised": s_o["days_impact_over_limit"]},
        "pump_unsetting_risk_days": {"baseline": s_b["days_unsetting_risk"],
                                     "optimised": s_o["days_unsetting_risk"]},
        "mean_pump_volumetric_efficiency": {"baseline": s_b["mean_vol_eff"], "optimised": s_o["mean_vol_eff"]},
        "field_oil_change_pct_same_steam_budget": report.get("field_steam_allocation", {}).get("field_oil_change_pct_vs_current"),
        "field_recovery_factor_10yr_pct": ({k: report["field_steam_allocation"][k]["mean_recovery_factor_10yr_pct"]
                                            for k in ["current_practice_12_2_180", "allocated_same_field_budget"]}
                                           if "field_steam_allocation" in report else None),
        "note": "Magnitudes come from synthetic, uncalibrated reservoir data -- recalibrate with "
                "Baghewala production history (fine_tune_reservoir) before field use."}
    report["runtime_s"] = round(time.time() - T0, 1)
    with open(os.path.join(a.out, "digital_twin_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    if not a.no_plots:
        make_plots(a.out, res_model, res_df, cards, df_opt, df_base, a.well, dev, ff_base, ff_rec)
    print(f"\nDone in {report['runtime_s']} s. Outputs in {a.out}")


def make_plots(out, res_model, res_df, cards, df_opt, df_base, well, dev, ff_base=None, ff_rec=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping plots"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    with torch.no_grad():
        for w, g in res_df.groupby("well_id"):
            wp = WELL_REGISTRY[w]
            tau = torch.tensor(g.t_days_since_production_start.values + wp["t_soak"],
                               dtype=torch.float32, device=dev).unsqueeze(1)
            pw_ = well_params_tensor(w, len(g), dev)
            pw_["rh"] = torch.tensor(g.heated_radius_m.values, dtype=torch.float32, device=dev).unsqueeze(1)
            o = res_model(tau, pw_)
            l = ax[0].plot(g.t_days_since_production_start, g.reservoir_temp_K, lw=3, alpha=0.3)[0]
            ax[0].plot(g.t_days_since_production_start, o["T"].cpu(), "--", c=l.get_color(), label=w)
            ax[1].plot(g.t_days_since_production_start, g.oil_rate_bbl_day, lw=3, alpha=0.3, c=l.get_color())
            ax[1].plot(g.t_days_since_production_start, o["Q_ref_bpd"].cpu(), "--", c=l.get_color(), label=w)
    ax[0].set(title="Reservoir temperature (thick=data, dashed=PINN)", xlabel="production day", ylabel="K")
    ax[1].set(title="Oil rate at reference drawdown", xlabel="production day", ylabel="bbl/d", yscale="log")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "reservoir_fit.png"), dpi=130); plt.close(fig)

    if ff_base is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for ff, lab, c in [(ff_base, "current 12/2/180", "tab:gray"), (ff_rec, "recommended", "tab:orange")]:
            cu = ff["_curves"]
            for i, key in enumerate(["T", "q"]):
                ax[i].fill_between(cu["t"], cu[key][0], cu[key][2], color=c, alpha=0.25)
                ax[i].plot(cu["t"], cu[key][1], c=c, label=f"{lab} (P50, band P10-P90)")
        ax[0].set(title="Baghewala: reservoir temperature", xlabel="production day", ylabel="K")
        ax[1].set(title="Baghewala: oil rate at field drawdown", xlabel="production day", ylabel="bbl/d")
        ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, "baghewala_field_forecast.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, len(cards), figsize=(4 * len(cards), 3.6))
    for a_, (cid, (g, card)) in zip(ax, cards.items()):
        a_.plot(g.surface_position_ft, g.surface_load_lbf, c="0.7", lw=3, label="surface data")
        a_.plot(card.surface_pos_ft, card.surface_load_lbf, "b--", label="surface PINN")
        a_.plot(g.pump_position_ft, g.pump_load_lbf, c="salmon", lw=3, label="pump data (unseen)")
        a_.plot(card.pump_pos_ft, card.pump_load_lbf, "r--", label="pump PINN")
        a_.set(title=cid, xlabel="position ft", ylabel="load lbf")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "srp_cards.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for df, lab in [(df_base, "fixed 8 SPM"), (df_opt, "optimised")]:
        ax[0, 0].plot(df.day, df.spm, label=lab); ax[0, 1].plot(df.day, df.fillage, label=lab)
        ax[1, 0].plot(df.day, df.rod_float_index, label=lab); ax[1, 1].plot(df.day, df.impact_lbf, label=lab)
    ax[0, 0].set(title=f"SPM ({well})"); ax[0, 1].set(title="pump fillage")
    ax[1, 0].axhline(1, c="k", ls=":"); ax[1, 0].set(title="rod-floating index (>=1 floats)", xlabel="day")
    ax[1, 1].axhline(FIELD_CONFIG["impact_allow_lbf"], c="k", ls=":")
    ax[1, 1].set(title="impact load at fluid contact, lbf", xlabel="day")
    ax[0, 0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "srp_operation.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
