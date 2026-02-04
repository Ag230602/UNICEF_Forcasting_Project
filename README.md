# 🌍 AI‑Driven, Uncertainty‑Aware Hurricane Forecasting & 3D Decision Support

**A UNICEF × UMKC Collaborative Research and Demonstration Repository**

This repository presents an end‑to‑end, governance‑safe system that couples **spatio‑temporal graph learning**, **uncertainty‑aware hurricane forecasting**, and **auditable 3D video visualization** to support **child‑centered humanitarian decision‑making** for disaster preparedness, response, and recovery.

The project is designed not as a black‑box prediction tool, but as a **transparent decision interface** that communicates *what may happen, how uncertain it is, and where limited resources should be prioritized*.

---
Data link :https://drive.google.com/drive/folders/1Qnki8L52euDNkveHjIheGZH-cz7gt1CB?usp=sharing

##  Project Objectives

* Forecast hurricane trajectories and impacts under uncertainty
* Quantify child exposure, vulnerability, and service disruption risk
* Translate probabilistic model outputs into **interpretable 3D visual narratives**
* Enable governance‑compliant, auditable decision support aligned with UNICEF operational needs

---

##  System Overview

**Core Components**

1. **Spatio‑Temporal Graph Learning (ST‑GNN)**
   Models storm evolution, hazard spread, vulnerability, and recovery dynamics over space and time.

2. **Uncertainty‑Aware Forecasting**
   Supports deterministic and probabilistic outputs (P10 / P50 / P90), enabling planning under plausible ranges rather than single outcomes.

3. **3D Visualization & Video Demonstration Layer**
   Converts model outputs into reproducible, frame‑based 3D videos used for planning briefings, training, and scenario comparison.

4. **Governance & Traceability Layer**
   Ensures reproducibility, metadata embedding, and policy‑safe communication.

---

## Demonstration Videos (YouTube)

> The standard approach is a clickable preview image that opens the video externally.

### Hurricane Irma (2017) — 3D Path Visualization
[![Hurricane Irma 3D Path](https://img.youtube.com/vi/ZvJ8jOmbHDE/hqdefault.jpg)](https://youtu.be/ZvJ8jOmbHDE)

### Uncertainty & Ensemble Spread Visualization
[![Uncertainty & Ensemble Spread](https://img.youtube.com/vi/nTIp0jjtJEk/hqdefault.jpg)](https://youtu.be/nTIp0jjtJEk)

### Recovery Rays & Impact Heatmap Demonstration
[![Recovery Rays & Impact Heatmap](https://img.youtube.com/vi/TCNdMnLFamw/hqdefault.jpg)](https://youtu.be/TCNdMnLFamw)


> These videos are **offline, pre‑rendered demonstrations** designed for interpretability and governance‑safe communication — not real‑time operational forecasts.

---

##  Repository Structure

```
repo/
│── src/
│   ├── data/          # Data loading, preprocessing, alignment
│   ├── modeling/      # ST‑GNN, trajectory models, forecasting heads
│   ├── evaluation/    # ADE/FDE, uncertainty metrics, diagnostics
│   ├── viz/           # 2D/3D visualization utilities
│   └── utils/         # Shared helpers
│
│── project_data/      # Plain‑text or structured domain data
│── outputs/
│   ├── metrics/
│   ├── figures/
│   └── videos/
│
│── requirements.txt
│── README.md
│── LICENSE
```

---

##  How to Run (Research / Demo Mode)

```bash
pip install -r requirements.txt
python src/main.py
```

> Note: Visualization demos are typically generated offline using deterministic frame pipelines.

---

##  Data Sources (Demonstration Case)

| Data Source | Provider                 | Purpose                                        |
| ----------- | ------------------------ | ---------------------------------------------- |
| HURDAT2     | NOAA                     | Historical storm tracks & timing               |
| ERA5        | ECMWF                    | Hazard context (wind, pressure, precipitation) |
| SVI 2022    | CDC / ATSDR              | Social vulnerability indicators                |
| Facilities  | ArcGIS / HDX             | Schools, hospitals, shelters                   |
| Geography   | State / Coastline layers | Spatial grounding                              |

**Important:**

* No raw satellite imagery is ingested
* Visuals are **stylized representations** derived from validated inputs

---

##  Graph Dataset Specification (Summary)

* **Nodes (N):** 5,122 census tracts
* **Edges (M):** 61,464 (k‑nearest neighbor graph)
* **Static Features:** Vulnerability + facility aggregates
* **Dynamic Features:** Time‑indexed hazard fields (ERA5 / ensembles)
* **Labels:** Recovery and impact indices

---

##  Evaluation & Diagnostics

**Forecast Accuracy**

* ADE / FDE @ 24h, 48h, 72h
* Observed vs predicted trajectory overlays

**Uncertainty Quality**

* P50 / P90 coverage
* Reliability and sharpness diagnostics

**Humanitarian Impact Metrics**

* Children exposed (P50 / P90)
* Child risk density index
* Schools & healthcare facilities at risk
* Service access degradation proxies

---

## 3D Visualization Methodology (High Level)

* Deterministic, frame‑based rendering pipeline
* Time‑synchronized storm motion and camera control
* Visual encodings for:

  * Storm path & uncertainty envelopes
  * Hazard intensity surfaces
  * Vulnerability and child exposure heatmaps
  * Recovery prioritization cues ("recovery rays")

Each frame embeds:

* Model version ID
* Data timestamp
* Ensemble size
* Uncertainty summary

This ensures **auditability and reproducibility**.

---

##  Limitations

* Not a real‑time operational forecasting system
* No physics‑based atmospheric simulation
* Visual encodings are qualitative and relative
* Intended for **decision support & communication**, not automated command

---

##  Planned Extensions

* Near‑real‑time data ingestion
* Interactive 2D/3D dashboards (time, viewpoint, scenarios)
* Ensemble‑aware visualization (P10 / P50 / P90)
* GenAI‑generated, citation‑grounded humanitarian briefings
* Cross‑region and cross‑scenario comparisons

---

##  UNICEF × UMKC Collaboration Roadmap

* Joint data governance & licensing framework
* Child‑centric recovery and exposure benchmarks
* Open evaluation protocols for humanitarian forecasting
* Field pilots, training workshops, and capacity building

---

##  License & Attribution

* Research code: MIT License (unless otherwise specified)
* Data: governed by original provider licenses (NOAA, ECMWF, CDC, etc.)

---

##  Acknowledgements

* **UNICEF** — operational guidance, child‑centered indicators
* **University of Missouri–Kansas City (UMKC)** — modeling, visualization, and research leadership
* NOAA, ECMWF, CDC/ATSDR, and open humanitarian data partners

---

**This repository is intended as a living research and demonstration platform for uncertainty‑aware, child‑centered humanitarian decision support.**



