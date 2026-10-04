import os
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np

def main():
    scada_dir = r"C:\Akash\PINN_AND_DATA\scada_pinn\baghewala_scada_twin\scada"
    origin_db = os.path.join(scada_dir, "advanced_ui", "data", "origin.db")
    srp_csv = os.path.join(scada_dir, "twin", "updated_results", "srp_daily_optimised.csv")
    css_csv = os.path.join(scada_dir, "twin", "reservoir_css_timeseries.csv")
    
    if not os.path.exists(srp_csv) or not os.path.exists(css_csv):
        print("Required CSV files not found.")
        return

    srp_df = pd.read_csv(srp_csv)
    css_df = pd.read_csv(css_csv)
    
    srp_df['day_int'] = srp_df['day'].astype(int)
    css_df['day_int'] = css_df['t_days_since_production_start'].astype(int)
    
    # Merge on day_int
    merged = pd.merge(srp_df, css_df, on='day_int', how='inner')
    
    records = []
    base_time = datetime.now() - timedelta(days=len(merged))
    
    for i, row in merged.iterrows():
        t = base_time + timedelta(days=i)
        
        temp_c = row.get('reservoir_temp_K_x', row.get('reservoir_temp_K_y', 320)) - 273.15
        if np.isnan(temp_c): temp_c = 47.0
        
        oil_rate = float(row.get('oil_rate_bpd', row.get('oil_rate_bbl_day', 0)))
        
        record = {
            "well_id": "BGW-01",
            "well_name": "Baghewala-01 (PINN Model)",
            "timestamp": t.isoformat() + "Z",
            "css_cycle_id": 1,
            "phase": 'production',
            "reservoir_temperature": float(temp_c),
            "reservoir_pressure": 10.0, # bar placeholder
            "oil_viscosity": float(row.get('viscosity_cP', 1000)),
            "oil_api": 18.0,
            "water_cut": 0.05,
            "steam_volume": 400.0, 
            "steam_rate": 0.0,
            "injection_pressure": 0.0,
            "injection_duration": 0.0,
            "soak_time": 0.0,
            "oil_rate_bopd": oil_rate,
            "stroke_length": float(row.get('stroke_in', 0) * 0.0254), # m
            "spm": float(row.get('spm', 0)),
            "vfd_setting": float(row.get('spm', 10) / 12.0 * 100), 
            "pump_efficiency": float(row.get('pump_fillage', 0.8)),
            "rod_load": float(row.get('pprl_lbf', 0) * 0.004448), # kN
            "energy_consumption": float(row.get('energy_kWh', 0)),
            "sor": 1.5,
            "energy_per_barrel": float(row.get('energy_kWh', 0) / max(1, oil_rate)),
            "rod_floating_probability": float(row.get('rod_float_index', 0)),
            "failure_probability": float(row.get('fluid_pound', 0) * 0.1), 
            "pump_unsetting_event": 1 if row.get('fluid_pound', 0) > 0 else 0,
            "rod_floating_event": 1 if row.get('rod_float_index', 0) > 1.0 else 0,
            "rod_failure_event": 0,
            "operating_cost": float(row.get('energy_kWh', 0) * 0.1),
            "cumulative_oil": float(row.get('cum_oil_bbl', 0)), 
            "raw_json": "{}"
        }
        records.append(record)
    
    conn = sqlite3.connect(origin_db)
    df = pd.DataFrame(records)
    
    conn.execute("DELETE FROM well_records WHERE well_id='BGW-01'")
    df.to_sql("well_records", conn, if_exists="append", index=False)
    conn.close()
    
    print(f"Successfully bridged {len(df)} records from pinn_digital_twin.py outputs into advanced_ui database.")

if __name__ == "__main__":
    main()
