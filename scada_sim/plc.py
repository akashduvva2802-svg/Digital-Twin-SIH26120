"""
Well-pad PLC / RTU logic. The PLC is the authority: the twin can only REQUEST set-points.

 * set-point validation: REMOTE mode, bounds, max step per write, allowed crank holes
 * pump-off controller settings (used by the plant model)
 * load interlocks: low-load alarm / trip (rod floating), high-load trip (structure)
 * heartbeat watchdog: if the twin goes silent, revert to local safe settings
"""
from .config import PLC


class PLCLogic:
    def __init__(self, cfg=None):
        self.cfg = dict(PLC if cfg is None else cfg)
        self.mode = 1                       # 1 REMOTE (twin requests accepted), 0 LOCAL
        self.alarm = 0
        self.alarm_text = ""
        self.last_reject = ""
        self.last_hb = None
        self.last_hb_t = 0.0
        self.low_load_count = 0
        self.events = []                    # (t, type, text)
        self.counters = {"trips_low_load": 0, "trips_high_load": 0, "low_load_alarms": 0,
                         "writes_accepted": 0, "writes_rejected": 0, "watchdog_fallbacks": 0}

    def log(self, t, kind, text):
        self.events.append((t, kind, text))
        self.alarm_text = text

    # ---------------------------------------------------------------- set-point requests
    def request(self, t, plant, spm=None, run_frac=None, stroke=None):
        c = self.cfg
        if self.mode != 1:
            self._reject(t, "PLC in LOCAL mode (watchdog fallback) -- request ignored"); return False
        ok = True
        if spm is not None and abs(spm - plant.spm) > 1e-6:
            if not (c["spm_min"] <= spm <= c["spm_max"]):
                self._reject(t, f"SPM {spm:.2f} outside [{c['spm_min']}, {c['spm_max']}]"); ok = False
            elif abs(spm - plant.spm) > c["spm_max_step"] + 1e-9:
                self._reject(t, f"SPM step {spm - plant.spm:+.2f} exceeds max step {c['spm_max_step']}"); ok = False
            else:
                plant.spm = float(spm); self.counters["writes_accepted"] += 1
                self.log(t, "SETPOINT", f"SPM -> {spm:.2f} (twin)")
        if run_frac is not None and abs(run_frac - plant.run_frac) > 1e-6:
            if not (c["run_frac_min"] <= run_frac <= 1.0):
                self._reject(t, f"run fraction {run_frac:.2f} outside [{c['run_frac_min']}, 1]"); ok = False
            else:
                plant.run_frac = float(run_frac); self.counters["writes_accepted"] += 1
                self.log(t, "SETPOINT", f"run fraction -> {run_frac:.2f} (twin)")
        if stroke is not None and abs(stroke - plant.stroke) > 1e-6:
            if stroke not in c["stroke_allowed_in"]:
                self._reject(t, f"stroke {stroke} in not an available crank hole"); ok = False
            else:
                self._reject(t, f"stroke change to {stroke} in needs a field crew: work order raised")
                ok = False
        return ok

    def _reject(self, t, why):
        self.last_reject = why
        self.counters["writes_rejected"] += 1
        self.log(t, "REJECT", why)

    # ---------------------------------------------------------------- heartbeat watchdog
    def heartbeat(self, t, hb):
        if hb is not None and hb != self.last_hb:
            self.last_hb, self.last_hb_t = hb, t
            if self.mode == 0 and self.alarm & 8:
                self.mode = 1; self.alarm &= ~8
                self.log(t, "WATCHDOG", "twin heartbeat restored -> REMOTE")

    def watchdog(self, t, plant):
        if self.mode == 1 and self.last_hb is not None and t - self.last_hb_t > self.cfg["watchdog_s"]:
            self.mode = 0; self.alarm |= 8
            self.counters["watchdog_fallbacks"] += 1
            if self.cfg.get("fallback", "hold_last") == "nameplate":
                plant.spm, plant.run_frac = self.cfg["default_spm"], self.cfg["default_run_frac"]
                self.log(t, "WATCHDOG", "twin heartbeat lost -> LOCAL, nameplate default set-points")
            else:
                # Hold the last ACCEPTED set-point: the nameplate default (8 SPM) proved unsafe in cold,
                # viscous oil -- in integration testing it caused both remaining rod-floating trips.
                self.log(t, "WATCHDOG", f"twin heartbeat lost -> LOCAL, holding last accepted set-point "
                                        f"({plant.spm:.2f} SPM, run {plant.run_frac:.2f})")

    # ---------------------------------------------------------------- load interlocks (per card)
    def check_card(self, t, plant, pprl, mprl):
        c = self.cfg
        W = plant.static_rod_weight()
        self.alarm &= ~3
        if pprl > c["high_load_trip_lbf"]:
            self.alarm |= 2 | 4
            plant.trip_left = c["trip_down_hours"] * 3600; self.counters["trips_high_load"] += 1
            self.log(t, "TRIP", f"high load {pprl:.0f} lbf > {c['high_load_trip_lbf']:.0f}")
            return
        if mprl < c["low_load_alarm_frac"] * W:
            self.alarm |= 1; self.counters["low_load_alarms"] += 1
            self.low_load_count = self.low_load_count + 1 if mprl < c["low_load_trip_frac"] * W else 0
            self.log(t, "ALARM", f"low load {mprl:.0f} lbf (< {c['low_load_alarm_frac']:.0%} of rod weight): rod floating risk")
            if self.low_load_count >= c["low_load_trip_cards"]:
                self.alarm |= 4
                plant.trip_left = c["trip_down_hours"] * 3600; self.counters["trips_low_load"] += 1
                self.low_load_count = 0
                self.log(t, "TRIP", f"rods floating: {c['low_load_trip_cards']} cards below trip level, well down "
                                    f"{c['trip_down_hours']:.0f} h")
        else:
            self.low_load_count = 0
