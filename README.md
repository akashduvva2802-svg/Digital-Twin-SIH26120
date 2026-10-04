# 🚀 Physics-Informed Digital Twin for Well-to-Surface Optimization (CSS & SRP)

**Smart India Hackathon 2026** | **Problem Statement ID:** 26120 | **Team:** Elite Exchangers
**Theme:** Smart Automation | **Target Field:** Baghewala Heavy Oil Field (Oil India Limited)

---

## 📖 Overview
This repository contains a complete **Coupled Physics-Informed Digital Twin** designed to optimize heavy oil extraction in Cyclic Steam Stimulation (CSS) wells equipped with Sucker Rod Pumps (SRP). 

Extracting heavy oil is highly energy-intensive and prone to severe mechanical failures. To solve this, we built a dual-scale digital twin that uses **Physics-Informed Neural Networks (PINNs)** to model macroscopic reservoir thermal decay and microscopic rod-string wave dynamics. The twin operates as an autonomous edge agent, integrating directly with a real-time SCADA environment via OPC UA to deliver closed-loop optimization.

---

## 💻 Tech Stack

Our solution bridges modern deep learning with industrial operational technology (OT) using a fully open-source Python stack:

*   **Frontend / Dashboard UI:** `Streamlit` (Real-time web application), `Plotly` (Interactive gauge, time-series, and dynamometer card visualizations).
*   **Backend & Concurrency:** `Python 3.11+`, `asyncio` (For non-blocking SCADA polling and continuous heartbeat watchdogs).
*   **Database / Historian:** `SQLite` (Configured in WAL mode for high-frequency concurrent read/writes between the SCADA server and Twin).
*   **Machine Learning (PINNs):** `PyTorch` (Multi-Layer Perceptrons, Cauchy solvers, and custom physics-residual loss functions).
*   **SCADA / OT Protocol:** `asyncua` (FreeOpcUa stack implementing the industrial OPC UA standard for real-time telemetry and setpoint control).
*   **Mathematical Optimization (CSS & SRP):** 
    *   *CSS Field Optimization:* `PyTorch Autograd` (Gradient descent through the Thermal PINN for steam-cycle length optimization).
    *   *SRP Real-Time Control:* `SciPy`, `NumPy` (Vectorized grid-search optimization bounded by mechanical torque/stress limits).

---

## 🧠 Physics-Informed Neural Networks (PINNs)

| Feature | **Reservoir Thermal & Flow PINN** (Macroscale) | **Sucker Rod Pump (SRP) Wave PINN** (Microscale) |
| :--- | :--- | :--- |
| **Primary Purpose** | Predict reservoir heating/cooling and oil production rates based on CSS parameters. | Translate noisy surface dynamometer cards into accurate hidden downhole pump cards. |
| **Mathematical Physics** | Marx-Langenheim (heating), Boberg-Lantz (cooling), Zonal Dupuy (inflow). | Everitt & Jennings damped-wave equation. |
| **Network Topology** | Multi-Layer Perceptron (MLP) with Tanh (~51,000 parameters). | Multi-Layer Perceptron (MLP) with Tanh (~46,000 parameters). |
| **Input Features** | 11 Features: Time embeddings + 8 physical properties ($R_h, h, K, \mu_{cold}$, etc.) | 20 Features: Spatial depth ($\xi$), periodic time (Fourier harmonics), and 7 rod properties. |

---

## 🏗️ Synthetic SCADA Environment + Real-Time Twin Integration

A self-contained plant-and-SCADA test bed, of the kind control engineers use to build and prove a digital twin before connecting it to a real field. The twin runs as a separate service and talks to SCADA only through OPC UA, exactly as it would in the field.

```text
 scada_sim/ (the "field")                                      twin/ (the digital twin)
 ┌──────────────────────────────────────────────┐   OPC UA    ┌──────────────────────────────┐
 │ plant.py   true well (hidden parameters,     │◄──────────► │ connector.py                  │
 │            annulus fluid level, wave-equation │  tags,      │  historian -> self-configure  │
 │            cards, motor, faults)              │  cards,     │  card -> PINN diagnosis       │
 │ plc.py     pump-off control, load trips,      │  set-point  │  -> state/parameter estimate  │
 │            bounds/ramp checks, watchdog       │  requests,  │  -> optimiser + card feedback │
 │ server.py  OPC UA server, scan loop, scenario │  heartbeat  │  -> approval policy -> write  │
 │ historian  SQLite: live tags, cards, events + │             │  -> readback -> audit trail   │
 │            3-yr seeded field history          │◄──── SQL ───│ pinn_digital_twin.py (PINNs)  │
 └──────────────────────────────────────────────┘             └──────────────────────────────┘
```

## Run

```bash
pip install asyncua torch numpy pandas scipy streamlit plotly
python run_integration.py            # ~3 min: seed history, baseline run, twin run, 9 acceptance tests
python run_integration.py --tests-only   # re-evaluate the last run
```
Manual: `python -m scada_sim.server --db field.db --days 5` in one terminal and
`python twin/connector.py --db field.db` in another. Any OPC UA client (e.g. UaExpert) can
browse `opc.tcp://127.0.0.1:4840/baghewala/scada/` while it runs.

## What each layer does

| Layer | File | Real-field equivalent |
|---|---|---|
| True well | `plant.py` | The actual well. Parameters are hidden from and deliberately different to the twin: 130 mD, 13 Pa.s, tubing at 60 % of reservoir excess temperature |
| PLC / RTU | `plc.py` | Well-pad controller. The **authority**: the twin can only request. Bounds 2–10 SPM, max step 1.5 SPM, crank-hole stroke changes need a work order, low/high-load trips, 3 h heartbeat watchdog, fallback = hold last accepted set-point |
| SCADA server | `server.py` | OPC UA server `Baghewala/BGW-01/{SRP,VFD,Wellhead,Steam,PLC,Twin}` with 37 tags (`config.py`), cards as 400-point arrays |
| Historian | `historian.py`, `history_seed.py` | Time-series store and records: production, well tests, CSS cycles, steam injection, VFD daily, rod failures, pump unsetting, completion, fluid properties, pressure surveys (10 wells × 3 years) |
| Twin connector | `twin/connector.py` | Integration service: configures from the historian, then heartbeat → card → diagnosis → estimation → decision → request → readback → audit |

## Twin decision loop (per new card)

1. **Diagnose** the surface card with the SRP PINN, warm-started from the previous card (~4.5 s).
2. **Estimate the state**: measured rod friction (mid-stroke load gap; verified within ~10 % in the floating regime), productivity correction from the daily allocated rate, water cut, and fluid level from echometer shots.
3. **Decide**:
   - Safety corrections from the measured card always win (floating → cut SPM; fluid pound → cut SPM or run time).
   - Otherwise it moves predictively toward the model optimum (`optimize_srp`). While the model is flagged uncertain, it only moves toward lower SPM.
   - On a suspect sensor it holds and raises an alert.
4. **Approve and write**: bounded by the ramp limit, written to `Twin/SPM_SP` and `Twin/RunFrac_SP`, then the PLC validates and the readback is confirmed.
5. **Audit**: every step is logged to the historian, and recommendations and alarms are mirrored to SCADA tags for the HMI.

## Acceptance tests (last run: 9 / 9 PASS, `integration_out/acceptance_report.json`)

Scenario, identical for both runs:
- day 1: inflow drops 55 %
- days 2–3: tubing cools and asphaltene friction builds up to ×6.5
- days 3.5–3.75: load-cell channel desynchronised
- day 4 (twin run only): the twin goes offline for 6 h

| Test | Result |
|---|---|
| I1 Every problem-statement data type in the historian; twin self-configures | PASS |
| I2 Live data path SCADA → OPC UA → twin | 118 cards, 21 diagnosed |
| I3 Out-of-bounds request rejected by PLC | "SPM 20.00 outside [2.0, 10.0]" |
| I4 Set-points applied within bounds and ramp, with readback | 21 accepted, 8 → 2.2 SPM in ≤1.5 steps |
| I5 Sensor fault detected, no write on bad data | PASS |
| I6 Twin outage → watchdog fallback in 2.9 h → REMOTE on recovery | PASS |
| I7 vs current practice | floating trips 10 → 0, kWh/bbl 3.10 → 2.19, impact 4.6 → 1.7 h, oil 58.6 → 59.0 bbl |
| I8 Real-time latency | median 4.5 s, max 14 s (cold start) |
| I9 Audit trail complete | PASS |

## Findings from integration testing

These are the reasons to test against a plant before the field:

- **Heartbeat must be its own task.** When it was sent from the decision loop, long diagnoses starved it, and the PLC fell back 22 times and accepted no twin set-points.
- **A nameplate fallback (8 SPM) is unsafe in cold oil.** It caused every remaining floating trip, so the fallback now holds the last accepted set-point.
- **Friction must be measured, not stepped.** A multiplicative estimator ran away (×10), turned good cards into "sensor faults", and froze the twin during the real floating event.
- **Viscosity rules must be consistent.** The connector ignored water while the optimiser included it, so every card was flagged "load too low" for two days.
- **Pure reaction is not enough.** Without the predictive move toward the model optimum, the twin sat at 8 SPM until the rods were already floating.

## Limitations

- The SCADA and the plant are **simulated**. A real deployment needs Oil India's tag list, network access through a DMZ, and OPC UA security with certificates (this test bed runs without encryption).
- SRP decisions use the day-by-day optimiser plus card feedback. The **MPC** (horizon prediction of fluid level, move suppression) is not built yet; `decide()` is where it plugs in.
- The plant and twin share textbook physics forms, with different parameters. Emulsion viscosity uses log-mixing, but heavy-oil emulsions below inversion are usually more viscous than oil. The online friction estimate absorbs part of that error.
- Time is accelerated (1 simulated hour per second) on one CPU core, and the twin diagnoses the newest card rather than every card.
- The failure history is synthetic, from a hazard model, so the twin reports it but no reliability model is trained yet.
- The twin copy in `twin/` is the verified twin plus an optional `cache=` warm-start argument in `diagnose_scada_card`. It does not include the unfinished steam-pressure work.
