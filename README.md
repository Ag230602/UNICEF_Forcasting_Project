# 🌪 STORM-CARE  
## Uncertainty-Aware Hurricane Forecasting & Child-Centered Decision Intelligence  
### Spatio-Temporal Graph Neural Networks for Humanitarian Planning

Dr. Yugyung Lee (Professor of Computer Science), Adrija Ghosh (CS Graduate Student)
School of Science and Engineering, University of Missouri–Kansas City

<p align="center">
  <b>Research Prototype | Probabilistic Forecasting | Human-Centered AI | Governance-Aware Visualization</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ST--GNN-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-ff4b4b)
![Geospatial](https://img.shields.io/badge/Geo-Spatial_Data-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

# 🌍 Executive Summary

**STORM-CARE (Storm-Focused Child-Centered Actionable Risk Engine)** is a research and demonstration framework that transforms hurricane forecast uncertainty into actionable humanitarian planning intelligence using:

- 📈 Spatio-Temporal Graph Neural Networks (ST-GNN)  
- 📊 Probabilistic forecasting (P10 / P50 / P90)  
- 🌍 Multi-source geospatial data fusion  
- 🧒 Child-centered vulnerability modeling  
- 🗺 Governance-aware visualization  

Unlike traditional hurricane systems that stop at meteorological prediction, STORM-CARE translates uncertainty into impact-aware decision signals.

Data link:https://drive.google.com/drive/folders/1Qnki8L52euDNkveHjIheGZH-cz7gt1CB?usp=drive_link

 
---

# 🎯 Problem Statement

Emergency planners need more than:

- "Where will the storm go?"

They need:

- How many children could be exposed?
- Which schools are at risk?
- Where will healthcare access degrade?
- What does worst-case uncertainty imply?

STORM-CARE bridges:

Meteorology → Uncertainty Modeling → Human Impact → Planning Intelligence


---

# 🏗 System Architecture

     Multi-Source Data Ingestion
                 ↓
     Geospatial Harmonization
                 ↓
 Feature Engineering (Static + Dynamic)
                 ↓
 Graph Construction (Tract-Level Nodes)
                 ↓
Spatio-Temporal Graph Neural Network
↓
Probabilistic Forecast Outputs (P10/P50/P90)
↓
Child-Centered Impact Translation
↓
Interactive Decision Dashboard


---

# 🌪 Core Technical Contributions

## 1️⃣ Spatio-Temporal Graph Forecasting

- Census tract–level node modeling  
- Spatial adjacency via k-nearest neighbor graphs  
- Temporal storm progression encoding  
- Vulnerability-weighted hazard diffusion  

---

## 2️⃣ Probabilistic Forecasting for Planning

Instead of a single deterministic track:

- **P50:** Median scenario  
- **P10:** Lower-bound scenario  
- **P90:** Upper-bound scenario  

This enables scenario-based resource allocation under uncertainty.

---

## 3️⃣ Child-Centered Impact Modeling

Derived indicators include:

- Children exposed (P50 & P90)  
- School disruption probability  
- Healthcare access degradation index  
- Infrastructure stress overlays  
- Recovery prioritization heatmaps  

---

# 🌍 Data Ecosystem

| Category | Dataset | Purpose |
|----------|----------|----------|
| Hurricane Tracks | NOAA HURDAT2 | Historical storm paths |
| Atmospheric Fields | ERA5 Reanalysis | Wind / pressure / precipitation |
| Social Vulnerability | CDC SVI (US) | Socioeconomic exposure proxy |
| Population | WorldPop | Demographic density |
| Infrastructure | HDX Facilities | Schools, hospitals, shelters |

---

## 🌎 Global Adaptation

For non-US deployment, vulnerability layers can be adapted using:

- INFORM Subnational Risk Index  
- DHS socio-economic indicators  
- World Bank poverty metrics  
- HDX humanitarian datasets  

---

# 📊 Evaluation Framework

## Deterministic Accuracy

- ADE – Average Displacement Error  
- FDE – Final Displacement Error  

Evaluated at 24h / 48h / 72h horizons.

---

## Probabilistic Quality

- Calibration analysis  
- Reliability diagrams  
- Sharpness vs dispersion trade-off  
- Empirical P50 / P90 coverage validation  

---

# 🖥 Interactive Demonstration

### 🌐 Streamlit Dashboard

https://stormcare-i9kz6hkvbpsydseiywfpqm.streamlit.app/

---

# 🎥 Visual Demonstrations  
*(Thumbnail links – GitHub friendly, no visible URLs)*

---

### 🌪 Hurricane Irma (2017) – 3D Track Visualization
[![Hurricane Irma 3D](https://img.youtube.com/vi/ZvJ8jOmbHDE/hqdefault.jpg)](https://www.youtube.com/watch?v=ZvJ8jOmbHDE)

---

### 📈 Ensemble Spread & Uncertainty Modeling
[![Uncertainty Spread](https://img.youtube.com/vi/nTIp0jjtJEk/hqdefault.jpg)](https://www.youtube.com/watch?v=nTIp0jjtJEk)

---

### 🔥 Recovery Rays & Impact Heatmaps
[![Recovery Visualization](https://img.youtube.com/vi/TCNdMnLFamw/hqdefault.jpg)](https://www.youtube.com/watch?v=TCNdMnLFamw)

---

# 🚀 Getting Started

```bash
git clone https://github.com/Ag230602/STORM_CARE.git
cd STORM_CARE

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
🗺 Visualization Capabilities
Interactive hazard heatmaps

Uncertainty cones (P50 centerline + P90 spread)

Child exposure overlays

Facility vulnerability markers

Recovery prioritization signals

🔐 Governance & Responsible Use
STORM-CARE:

Is a research prototype

Is not a real-time operational forecasting authority

Supports human decision-making

Requires responsible interpretation

Requires ethical handling of vulnerability data

🔬 Research Positioning
This project demonstrates:

Advanced spatio-temporal modeling

Graph neural networks in disaster forecasting

Probabilistic impact translation

Responsible AI for humanitarian systems

Human-centered uncertainty visualization

Suitable for:

AI + Climate research tracks

Humanitarian AI grants

Responsible AI initiatives

Computational sustainability conferences

🛣 Future Directions
Storm surge modeling integration

SAR-based flood segmentation (U-Net)

Real-time ensemble ingestion

Multi-hazard extension (flood + cyclone + heat)

Cross-country vulnerability harmonization
