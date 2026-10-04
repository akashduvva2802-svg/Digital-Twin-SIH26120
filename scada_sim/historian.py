"""
Historian (SQLite, WAL mode so the SCADA server and the twin can use it concurrently).

Live tables  : tag_history, cards, events (PLC events and twin audit trail)
Record tables: production_daily, well_tests, css_cycles, steam_injection_daily, vfd_daily,
               rod_failures, pump_unsetting, completion, fluid_properties, pressure_surveys
(these are the data types the problem statement says Baghewala has)
"""
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS tag_history (t REAL, well TEXT, tag TEXT, value REAL);
CREATE INDEX IF NOT EXISTS ix_tag ON tag_history(well, tag, t);
CREATE TABLE IF NOT EXISTS cards (t REAL, well TEXT, card_id INTEGER, spm REAL, stroke REAL,
                                  pos TEXT, load TEXT);
CREATE TABLE IF NOT EXISTS events (t REAL, well TEXT, source TEXT, kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS production_daily (date TEXT, well TEXT, oil_bpd REAL, water_bpd REAL,
                                  runtime_frac REAL, downtime_reason TEXT);
CREATE TABLE IF NOT EXISTS well_tests (date TEXT, well TEXT, oil_bpd REAL, water_bpd REAL,
                                  wellhead_T_degC REAL, test_hours REAL);
CREATE TABLE IF NOT EXISTS css_cycles (well TEXT, cycle INTEGER, inj_start TEXT, inj_days REAL,
                                  soak_days REAL, prod_days REAL, steam_t REAL, inj_pressure_psi REAL,
                                  steam_quality REAL, cycle_oil_bbl REAL);
CREATE TABLE IF NOT EXISTS steam_injection_daily (date TEXT, well TEXT, rate_tpd REAL,
                                  wellhead_pressure_psi REAL, quality REAL, fuel_mscf REAL);
CREATE TABLE IF NOT EXISTS vfd_daily (date TEXT, well TEXT, avg_hz REAL, avg_spm REAL, stroke_in REAL,
                                  kwh REAL, runtime_frac REAL, avg_current_A REAL);
CREATE TABLE IF NOT EXISTS rod_failures (date TEXT, well TEXT, depth_ft REAL, mode TEXT,
                                  spm_before REAL, wellhead_T_degC REAL, downtime_days REAL);
CREATE TABLE IF NOT EXISTS pump_unsetting (date TEXT, well TEXT, days_low_fillage_prior REAL,
                                  spm_before REAL, downtime_days REAL);
CREATE TABLE IF NOT EXISTS completion (well TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS fluid_properties (well TEXT PRIMARY KEY, api REAL, visc_cP_50C REAL,
                                  asphaltene_wt_pct REAL, visc_table TEXT);
CREATE TABLE IF NOT EXISTS pressure_surveys (date TEXT, well TEXT, static_bhp_psi REAL, datum_m REAL);
"""


class Historian:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def log_tags(self, t, well, values):
        self.db.executemany("INSERT INTO tag_history VALUES (?,?,?,?)",
                            [(t, well, k, float(v)) for k, v in values.items()])

    def log_card(self, t, well, card_id, spm, stroke, pos, load):
        self.db.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?)",
                        (t, well, card_id, spm, stroke, json.dumps([round(float(x), 4) for x in pos]),
                         json.dumps([round(float(x), 1) for x in load])))

    def log_event(self, t, well, source, kind, text):
        self.db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (t, well, source, kind, text))

    def commit(self):
        self.db.commit()

    def query(self, sql, args=()):
        return self.db.execute(sql, args).fetchall()

    def close(self):
        self.db.commit(); self.db.close()
