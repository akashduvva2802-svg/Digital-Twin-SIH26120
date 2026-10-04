"""
Synthetic Baghewala SCADA environment -- configuration and tag map.

The tag map follows a typical oilfield SCADA hierarchy:
    Baghewala / <well> / {SRP, VFD, Wellhead, Steam, Reservoir, PLC, Twin}
Engineering units are field units (ft, lbf, psi, bbl/d, degC) as a real historian would hold.
"""
import math

ENDPOINT = "opc.tcp://127.0.0.1:4840/baghewala/scada/"
NAMESPACE = "urn:baghewala:scada"
LIVE_WELL = "BGW-01"

# ---- time --------------------------------------------------------------------
SCAN_S = 60.0                 # plant integration step (simulated seconds)
CARD_INTERVAL_S = 3600.0      # a dynamometer card is captured every simulated hour
DEFAULT_ACCEL = 3600.0        # simulated seconds per real second (1 sim-hour per second)

# ---- completion (would come from the well-completion database) ---------------
COMPLETION = {
    "BGW-01": {"pump_depth_m": 1050.0, "rod_diam_in": 0.875, "rod_grade": "D", "rod_E_psi": 30.5e6,
               "rod_density_lbm_ft3": 490.0, "plunger_in": 1.5, "tubing_id_in": 2.441,
               "casing_id_in": 6.276, "tubing_od_in": 2.875, "stroke_holes_in": [54, 64, 74, 86],
               "unit": "C-228D-213-86", "structure_lbf": 21300, "gearbox_inlb": 228000,
               "vit_insulated_tubing": True, "spm_at_60Hz": 8.0},
}

# ---- TRUE plant parameters (hidden from the twin) ------------------------------
# Deliberately different from the twin's P50 well so the twin has to discover the well.
TRUE_WELL = {
    "BGW-01": {"h": 18.0, "k_md": 130.0, "mu_cold": 13.0, "mu_hot": 0.35, "T0": 320.5, "Ts": 523.0,
               "t_inj": 12.0, "t_soak": 2.0, "tubing_temp_fraction": 0.60, "api": 18.0,
               "asphaltene_wt_pct": 9.0, "p_reservoir_psi": 1378.0, "casing_pressure_psi": 45.0,
               "productivity_scale": 1.0, "rod_friction_scale": 1.0},
}

# ---- PLC / RTU settings ----------------------------------------------------------
PLC = {
    "spm_min": 2.0, "spm_max": 10.0, "spm_max_step": 1.5,        # bounds + max change per write
    "run_frac_min": 0.25, "stroke_allowed_in": [54, 64, 74, 86],
    "default_spm": 8.0, "default_stroke_in": 86.0, "default_run_frac": 1.0,   # local (fallback) settings
    "poc_fillage_sp": 0.70, "poc_idle_min": 30.0,                 # pump-off controller
    "low_load_alarm_frac": 0.15, "low_load_trip_frac": 0.02,      # x buoyed rod weight
    "low_load_trip_cards": 2, "high_load_trip_lbf": 21300.0, "trip_down_hours": 4.0,
    "watchdog_s": 3 * 3600.0,                                     # twin heartbeat timeout (sim s)
    "fallback": "hold_last",                                      # or "nameplate" (default_spm)
}

# ---- tag list: (path, engineering unit, writable, description) ------------------
TAGS = [
    # SRP / rod pump controller
    ("SRP/SPM", "1/min", False, "measured strokes per minute"),
    ("SRP/StrokeLength", "in", False, "current crank-hole stroke"),
    ("SRP/PumpFillage_POC", "-", False, "pump-off controller fillage estimate"),
    ("SRP/PPRL", "lbf", False, "peak polished-rod load, last card"),
    ("SRP/MPRL", "lbf", False, "minimum polished-rod load, last card"),
    ("SRP/CardID", "-", False, "increments on each new dynamometer card"),
    ("SRP/CardTimestamp", "s", False, "simulated time of last card"),
    ("SRP/Card_Position", "ft", False, "surface position array (400)"),
    ("SRP/Card_Load", "lbf", False, "surface load array (400)"),
    ("SRP/CardSPM", "1/min", False, "SPM when the last card was captured"),
    ("SRP/CardStroke", "in", False, "stroke when the last card was captured"),
    ("SRP/RunStatus", "-", False, "1 running, 0 idle (POC), -1 tripped"),
    ("SRP/RunTimeFrac_24h", "-", False, "running fraction over last 24 h"),
    # VFD / motor
    ("VFD/Frequency", "Hz", False, "drive output frequency"),
    ("VFD/MotorCurrent", "A", False, "motor current"),
    ("VFD/MotorPower", "kW", False, "motor input power"),
    ("VFD/Energy_24h", "kWh", False, "energy over last 24 h"),
    # wellhead / surveys
    ("Wellhead/TubingPressure", "psi", False, "tubing head pressure"),
    ("Wellhead/CasingPressure", "psi", False, "casing head pressure"),
    ("Wellhead/Temperature", "degC", False, "produced-fluid wellhead temperature"),
    ("Wellhead/FluidLevel_Acoustic", "ft", False, "fluid level above pump, last echometer shot"),
    ("Wellhead/FluidLevel_Timestamp", "s", False, "time of last echometer shot"),
    ("Wellhead/LiquidRate_Allocated", "bbl/d", False, "allocated liquid rate (daily)"),
    # steam
    ("Steam/Phase", "-", False, "0 production, 1 injection, 2 soak"),
    ("Steam/DaysSinceInjectionEnd", "d", False, "time since end of last injection"),
    ("Steam/InjectedThisCycle", "t", False, "steam injected in current cycle"),
    # PLC
    ("PLC/Mode", "-", False, "0 LOCAL, 1 REMOTE (twin set-points accepted)"),
    ("PLC/Alarm", "-", False, "bitfield: 1 low-load, 2 high-load, 4 trip, 8 watchdog"),
    ("PLC/AlarmText", "-", False, "last alarm message"),
    ("PLC/LastRejectReason", "-", False, "why the last set-point write was rejected"),
    ("PLC/SimTime", "s", False, "simulated plant time"),
    # set-points written by the twin (validated by the PLC)
    ("Twin/SPM_SP", "1/min", True, "SPM set-point request"),
    ("Twin/RunFrac_SP", "-", True, "pump-off controller run-time fraction request"),
    ("Twin/Stroke_SP", "in", True, "stroke change request (crank hole, needs field crew)"),
    ("Twin/Heartbeat", "-", True, "twin heartbeat counter"),
    ("Twin/Recommendation", "-", True, "latest twin recommendation text"),
    ("Twin/AlarmText", "-", True, "twin-raised alarm"),
]


def annulus_area_m2(c):
    return math.pi / 4 * ((c["casing_id_in"] * 0.0254) ** 2 - (c["tubing_od_in"] * 0.0254) ** 2)
