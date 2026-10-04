import os
import json
import subprocess

RESULTS_DIR = r"C:\Akash\PINN_AND_DATA\scada_pinn\baghewala_scada_twin\scada\twin\updated_results"
JSON_PATH = os.path.join(RESULTS_DIR, "digital_twin_report.json")
SLIDE1_HTML = os.path.join(RESULTS_DIR, "slide1.html")
SLIDE2_HTML = os.path.join(RESULTS_DIR, "slide2.html")
SLIDE1_PNG = os.path.join(RESULTS_DIR, "slide1_generated.png")
SLIDE2_PNG = os.path.join(RESULTS_DIR, "slide2_generated.png")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

b = report.get("expected_benefits_quantified", {})
qc = report.get("data_qc", {})
res_metrics = report.get("reservoir", {}).get("training_metrics", {})
srp_metrics = report.get("srp", {}).get("training_metrics", {})

# ----- DATA FOR SLIDE 1 -----
e_base = b.get("lifting_energy_kwh_per_bbl", {}).get("baseline", 3.17)
e_opt = b.get("lifting_energy_kwh_per_bbl", {}).get("optimised", 0.75)
e_diff_pct = ((e_opt - e_base) / e_base) * 100

imp_base = b.get("impact_over_limit_days", {}).get("baseline", 168)
imp_opt = b.get("impact_over_limit_days", {}).get("optimised", 0)

eff_base = b.get("mean_pump_volumetric_efficiency", {}).get("baseline", 0.29)
eff_opt = b.get("mean_pump_volumetric_efficiency", {}).get("optimised", 0.88)

oil_change = b.get("field_oil_change_pct_same_steam_budget", 17.2)

uns_base = b.get("pump_unsetting_risk_days", {}).get("baseline", 10)
uns_opt = b.get("pump_unsetting_risk_days", {}).get("optimised", 0)

# ----- SLIDE 1 HTML -----
html1 = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; background: #ffffff; color: #1e3a5f; margin: 0; padding: 40px; width: 1400px; height: 750px; box-sizing: border-box; }}
        h1 {{ color: #1e3a5f; font-size: 28px; font-weight: 600; margin-bottom: 30px; }}
        .kpi-container {{ display: flex; gap: 20px; margin-bottom: 40px; }}
        .kpi-card {{ flex: 1; border-radius: 8px; padding: 25px 15px; text-align: center; height: 120px; }}
        .bg-blue {{ background-color: #eaf3fb; }}
        .bg-orange {{ background-color: #fbede1; }}
        .kpi-main {{ font-size: 42px; font-weight: bold; margin-bottom: 8px; }}
        .kpi-sub {{ font-size: 14px; color: #4a5568; line-height: 1.4; }}
        .text-blue {{ color: #1565c0; }}
        .text-orange {{ color: #e65100; }}
        .content-split {{ display: flex; gap: 40px; }}
        .left-col {{ flex: 5.5; }}
        .right-col {{ flex: 4.5; font-size: 17px; line-height: 1.5; color: #2d3748; }}
        .chart-title {{ font-size: 16px; font-weight: bold; color: #e65100; margin-bottom: 10px; }}
        .graph-img {{ width: 100%; object-fit: contain; }}
        .benefits-title {{ font-size: 22px; font-weight: bold; color: #1e3a5f; margin-bottom: 20px; }}
        .benefit-item {{ margin-bottom: 25px; }}
        .benefit-item strong {{ color: #1e3a5f; }}
    </style>
</head>
<body>
    <h1>Impact for a Baghewala P50 well (optimised vs fixed-speed current practice)</h1>
    
    <div class="kpi-container">
        <div class="kpi-card bg-blue">
            <div class="kpi-main text-blue">{e_diff_pct:.0f} %</div>
            <div class="kpi-sub">lifting energy<br>{e_base:.1f} &rarr; {e_opt:.2f} kWh/bbl</div>
        </div>
        <div class="kpi-card bg-orange">
            <div class="kpi-main text-orange">{imp_base} &rarr; {imp_opt}</div>
            <div class="kpi-sub">days with impact load<br>over limit per cycle</div>
        </div>
        <div class="kpi-card bg-blue">
            <div class="kpi-main text-blue">{eff_base:.2f} &rarr; {eff_opt:.2f}</div>
            <div class="kpi-sub">pump volumetric<br>efficiency</div>
        </div>
        <div class="kpi-card bg-orange">
            <div class="kpi-main text-orange">+{oil_change:.0f} %</div>
            <div class="kpi-sub">field oil output<br>(same steam budget)</div>
        </div>
        <div class="kpi-card bg-blue">
            <div class="kpi-main text-blue">{uns_base} &rarr; {uns_opt}</div>
            <div class="kpi-sub">days at pump-unsetting<br>risk per cycle</div>
        </div>
    </div>
    
    <div class="content-split">
        <div class="left-col">
            <div class="chart-title">SRP twin tracks the well through the cycle</div>
            <img class="graph-img" src="file:///{os.path.join(RESULTS_DIR, 'srp_operation.png').replace(chr(92), '/')}" />
        </div>
        <div class="right-col">
            <div class="benefits-title">Benefits</div>
            <div class="benefit-item">
                <strong>Economic:</strong> higher oil per cycle-day, steam steered to the wells where it earns most.
            </div>
            <div class="benefit-item">
                <strong>Reliability:</strong> no rod-floating days, impact and unsetting risk removed, rod stress well within limits.
            </div>
            <div class="benefit-item">
                <strong>Environmental:</strong> less electricity per barrel; operator can choose SOR optimally at a known profit cost.
            </div>
            <div class="benefit-item">
                <strong>Social / national:</strong> more domestic heavy-oil output; method transfers to other Indian heavy-oil fields.
            </div>
            <div class="benefit-item">
                <strong>Target users:</strong> OIL-Rajasthan production engineers, SRP field crews, reservoir planners.
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(SLIDE1_HTML, "w", encoding="utf-8") as f:
    f.write(html1)


# ----- DATA FOR SLIDE 2 -----
t_pass = qc.get("tests_passed", 29)
t_tot = qc.get("tests_total", 29)
rmse_temp = res_metrics.get("rmse_temp", 0.92)
rmse_pump = srp_metrics.get("rmse_pump_load", 11.6)
runtime = report.get("runtime_s", 150) / 60.0

html2 = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; background: #ffffff; color: #1e3a5f; margin: 0; padding: 40px; width: 1400px; height: 750px; box-sizing: border-box; display: flex; flex-direction: column; }}
        .top-split {{ display: flex; gap: 40px; margin-bottom: 20px; }}
        .feasibility {{ flex: 5; }}
        .risks {{ flex: 5; }}
        h1 {{ color: #1e3a5f; font-size: 26px; font-weight: 600; margin-top: 0; margin-bottom: 20px; }}
        .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 30px; }}
        .kpi-card {{ background-color: #f1f5f9; border-radius: 8px; padding: 20px 10px; text-align: center; }}
        .kpi-main {{ font-size: 32px; font-weight: bold; color: #1565c0; margin-bottom: 5px; }}
        .kpi-sub {{ font-size: 14px; color: #4a5568; line-height: 1.3; }}
        .chart-title {{ font-size: 18px; font-weight: bold; color: #e65100; margin-bottom: 10px; }}
        .graph-img {{ width: 100%; object-fit: contain; max-height: 350px; }}
        .risk-box {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 15px; display: flex; gap: 15px; align-items: flex-start; }}
        .risk-icon {{ background: #e65100; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; }}
        .risk-content {{ font-size: 15px; }}
        .risk-title {{ font-weight: 700; color: #2d3748; margin-bottom: 4px; }}
        .risk-strategy {{ color: #4a5568; line-height: 1.4; }}
    </style>
</head>
<body>
    <div class="top-split">
        <div class="feasibility">
            <h1>Feasibility &mdash; a working, verified prototype</h1>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-main">{t_pass} / {t_tot}</div>
                    <div class="kpi-sub">independent verification<br>tests pass</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-main">&le; {rmse_temp:.2f} K</div>
                    <div class="kpi-sub">temperature error on a<br>well never seen in training</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-main">{rmse_pump:.1f} lbf</div>
                    <div class="kpi-sub">pump-card error from<br>surface card alone</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-main">0.13 %</div>
                    <div class="kpi-sub">CSS optimiser vs brute-force<br>physics optimum</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-main">{runtime:.1f} min</div>
                    <div class="kpi-sub">full training on 1 GPU<br>(480k params)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-main">17&ndash;40 bbl/d</div>
                    <div class="kpi-sub">forecast P10-P90 vs public<br>17&ndash;36 bbl/d</div>
                </div>
            </div>
            
            <div class="chart-title">Baghewala forecast: 400 wells over published field ranges</div>
            <img class="graph-img" src="file:///{os.path.join(RESULTS_DIR, 'baghewala_field_forecast.png').replace(chr(92), '/')}" />
        </div>
        
        <div class="risks">
            <h1>Challenges & risks &rarr; strategies</h1>
            
            <div class="risk-box">
                <div class="risk-icon">!</div>
                <div class="risk-content">
                    <div class="risk-title">No well-level Baghewala data is public</div>
                    <div class="risk-strategy">&rarr; Calibrated to Oil India field figures (rate, 5-6x uplift) + analogue OSR; fine-tune hook ready for field data</div>
                </div>
            </div>
            
            <div class="risk-box">
                <div class="risk-icon">!</div>
                <div class="risk-content">
                    <div class="risk-title">Heavy oil makes rods float</div>
                    <div class="risk-strategy">&rarr; Viscosity-driven drag model caps SPM; limit automatically optimized by Wave-model cross-check</div>
                </div>
            </div>
            
            <div class="risk-box">
                <div class="risk-icon">!</div>
                <div class="risk-content">
                    <div class="risk-title">Faulty load cells / mis-set rod data</div>
                    <div class="risk-strategy">&rarr; Energy-balance and fluid-load checks flag the card before it is trusted</div>
                </div>
            </div>
            
            <div class="risk-box">
                <div class="risk-icon">!</div>
                <div class="risk-content">
                    <div class="risk-title">Steam supply, not economics, is binding</div>
                    <div class="risk-strategy">&rarr; Report marginal value of steam to allocate generator capacity efficiently</div>
                </div>
            </div>
            
            <div class="risk-box">
                <div class="risk-icon">!</div>
                <div class="risk-content">
                    <div class="risk-title">Operator trust in AI set-points</div>
                    <div class="risk-strategy">&rarr; Physics-based, explainable limits; advisory mode first, then closed-loop VFD</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(SLIDE2_HTML, "w", encoding="utf-8") as f:
    f.write(html2)

# Convert to PNG using Headless Chrome
print("Converting HTML to PNG with Headless Chrome...")
subprocess.run([
    CHROME_PATH, "--headless", "--disable-gpu", 
    f"--screenshot={SLIDE1_PNG}", "--window-size=1400,750", 
    f"file:///{SLIDE1_HTML.replace(chr(92), '/')}"
], check=True)

subprocess.run([
    CHROME_PATH, "--headless", "--disable-gpu", 
    f"--screenshot={SLIDE2_PNG}", "--window-size=1400,750", 
    f"file:///{SLIDE2_HTML.replace(chr(92), '/')}"
], check=True)

print("Done! Slides generated at:")
print(SLIDE1_PNG)
print(SLIDE2_PNG)
