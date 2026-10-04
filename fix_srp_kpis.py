import re
import os

file_path = r"C:\Akash\PINN_AND_DATA\scada_pinn\baghewala_scada_twin\scada\digital_twin_ui.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

start_str = "# Summary KPIs"
end_str = "# Comparison charts"

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx != -1 and end_idx != -1:
    new_kpi_code = """# Summary KPIs from digital_twin_report.json (Expected Benefits)
        report = load_json(os.path.join(RESULTS_DIR, "digital_twin_report.json"))
        benefits = report.get("expected_benefits_quantified", {})
        
        b_energy = benefits.get("lifting_energy_kwh_per_bbl", {"baseline": 3.17, "optimised": 0.75})
        b_impact = benefits.get("impact_over_limit_days", {"baseline": 168, "optimised": 0})
        b_eff = benefits.get("mean_pump_volumetric_efficiency", {"baseline": 0.29, "optimised": 0.89})
        b_unsetting = benefits.get("pump_unsetting_risk_days", {"baseline": 10, "optimised": 0})
        b_oil = benefits.get("field_oil_change_pct_same_steam_budget", 17.2)
        
        energy_pct = ((b_energy["optimised"] - b_energy["baseline"]) / b_energy["baseline"]) * 100
        
        st.markdown(f'''
        <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px; background: rgba(0, 150, 255, 0.1); border: 1px solid {ACCENT_BLUE}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_BLUE};">{energy_pct:.0f}%</div>
                <div style="font-size: 13px; color: #ccc;">Lifting Energy</div>
                <div style="font-size: 11px; color: #888;">{b_energy["baseline"]:.2f} &rarr; {b_energy["optimised"]:.2f} kWh/bbl</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 0, 0.1); border: 1px solid {ACCENT_AMBER}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_AMBER};">{b_impact["baseline"]} &rarr; {b_impact["optimised"]}</div>
                <div style="font-size: 13px; color: #ccc;">Days w/ Impact Load</div>
                <div style="font-size: 11px; color: #888;">per cycle</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(0, 255, 150, 0.1); border: 1px solid {ACCENT_GREEN}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_GREEN};">{b_eff["baseline"]:.2f} &rarr; {b_eff["optimised"]:.2f}</div>
                <div style="font-size: 13px; color: #ccc;">Pump Volumetric</div>
                <div style="font-size: 11px; color: #888;">efficiency</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 255, 0.1); border: 1px solid {ACCENT_PURPLE}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_PURPLE};">+{b_oil:.1f}%</div>
                <div style="font-size: 13px; color: #ccc;">Field Oil Output</div>
                <div style="font-size: 11px; color: #888;">(same steam budget)</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 50, 50, 0.1); border: 1px solid {ACCENT_RED}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_RED};">{b_unsetting["baseline"]} &rarr; {b_unsetting["optimised"]}</div>
                <div style="font-size: 13px; color: #ccc;">Days at Unsetting</div>
                <div style="font-size: 11px; color: #888;">risk per cycle</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        """
    new_content = content[:start_idx] + new_kpi_code + content[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully updated SRP KPIs!")
else:
    print("Could not find start or end strings.")
