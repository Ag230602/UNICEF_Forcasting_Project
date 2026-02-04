# app_unified_dashboard.py
# -----------------------------------------------------------------------------
# Unified Decision Dashboard: Path → Risk → Actions → Recovery
# (Streamlit app designed for non-technical users)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app_unified_dashboard.py
#
# Data (CSV) expected (names can vary; columns will be auto-detected):
#   1) Track predictions (path):
#      - lead_hours (or lead/lead_h/lead_time)
#      - pred_lat / pred_lon (or mu_lat/mu_lon, or lat/lon)
#      - (optional) sigma_lat / sigma_lon  (uncertainty)
#
#   2) Facilities (optional):
#      - shelters.csv, hospitals.csv, schools.csv with lat/lon (or latitude/longitude)
#      - Optional: hospitals beds column (beds), shelters capacity/current_load
#      - Optional: schools children_est or capacity
#
#   3) Areas grid (for recovery + optional vulnerability):
#      - areas.csv with: area_id (or node_id/cell_id/FIPS) and lat/lon
#
#   4) Recovery predictions:
#      - recovery_predictions.csv with: area_id (or node_id/cell_id/FIPS), t, pred_recovery
#        (pred_recovery should be in [0,1] where 1=fast/better recovery)
#
# NOTE:
# - The UI intentionally avoids technical wording (no "nodes", "graph", "GNN", etc.).
# - Heatmaps are explained in plain language with an embedded legend.
# -----------------------------------------------------------------------------

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
# import google.generativeai as genai


import numpy as np
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    go = None
    _HAS_PLOTLY = False
import pandas as pd
import pydeck as pdk
import streamlit as st

# with st.expander(" Gemini health check"):
#     try:
#         import google.generativeai as genai
#         k = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_key_input")
#         if not k:
#             st.warning("No Gemini API key set.")
#         else:
#             genai.configure(api_key=k)
#             m = genai.GenerativeModel("gemini-1.5-flash")
#             r = m.generate_content("Say OK in one word.")
#             st.success(f"Gemini response: {r.text}")
#     except Exception as e:
#         st.error(f"Gemini failed: {e}")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class CFG:
    default_center: Tuple[float, float] = (27.5, -82.5)  # Florida-ish
    default_zoom: int = 5
    refresh_seconds: int = 60

    # Visual scaling for uncertainty (approx) for 2D confidence regions
    p50_scale: float = 1.177
    p90_scale: float = 2.146

cfg = CFG()

# -----------------------------------------------------------------------------
# Helpers: robust loading + column normalization
# -----------------------------------------------------------------------------
def _try_read_csv(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None

def normalize_latlon(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Accept lat/lon in common variants; return df with columns ['lat','lon'] and cleaned rows."""
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    cols = {c.lower(): c for c in out.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    lat_col = pick("lat", "latitude")
    lon_col = pick("lon", "lng", "longitude")
    x_col = pick("x")
    y_col = pick("y")
    ll_col = pick("longlat")

    if lat_col and lon_col:
        out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    elif x_col and y_col:
        out["lat"] = pd.to_numeric(out[y_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[x_col], errors="coerce")
    elif ll_col:
        s = out[ll_col].astype(str).str.replace(r"[()]", "", regex=True)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] == 2:
            out["lon"] = pd.to_numeric(parts[0], errors="coerce")
            out["lat"] = pd.to_numeric(parts[1], errors="coerce")
        else:
            out["lat"] = np.nan
            out["lon"] = np.nan
    else:
        # keep existing if already present
        if "lat" not in out.columns: out["lat"] = np.nan
        if "lon" not in out.columns: out["lon"] = np.nan

    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out

def ensure_area_id(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Create a user-friendly 'area_id' key without calling it a node."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "area_id" not in out.columns:
        if "node_id" in out.columns:
            out["area_id"] = out["node_id"].astype(str)
        elif "cell_id" in out.columns:
            out["area_id"] = out["cell_id"].astype(str)
        elif "FIPS" in out.columns:
            out["area_id"] = out["FIPS"].astype(str)
        else:
            out["area_id"] = [f"area_{i}" for i in range(len(out))]
    out["area_id"] = out["area_id"].astype(str)
    return out

def normalize_track_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize track CSV into columns: lead_hours, mu_lat, mu_lon, sigma_lat, sigma_lon."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    lead_col = pick(["lead_hours", "lead_h", "lead", "lead_time", "horizon_hours"])
    if lead_col is None:
        d["lead_hours"] = np.arange(len(d)) * 6
    elif lead_col != "lead_hours":
        d = d.rename(columns={lead_col: "lead_hours"})

    mu_lat_col = pick(["mu_lat", "pred_lat", "lat_pred", "lat", "latitude"])
    mu_lon_col = pick(["mu_lon", "pred_lon", "lon_pred", "lon", "longitude"])
    if mu_lat_col is None or mu_lon_col is None:
        return None
    if mu_lat_col != "mu_lat":
        d = d.rename(columns={mu_lat_col: "mu_lat"})
    if mu_lon_col != "mu_lon":
        d = d.rename(columns={mu_lon_col: "mu_lon"})

    sig_lat_col = pick(["sigma_lat", "std_lat", "lat_sigma"])
    sig_lon_col = pick(["sigma_lon", "std_lon", "lon_sigma"])
    if sig_lat_col is None:
        d["sigma_lat"] = np.nan
    elif sig_lat_col != "sigma_lat":
        d = d.rename(columns={sig_lat_col: "sigma_lat"})
    if sig_lon_col is None:
        d["sigma_lon"] = np.nan
    elif sig_lon_col != "sigma_lon":
        d = d.rename(columns={sig_lon_col: "sigma_lon"})

    for c in ["lead_hours", "mu_lat", "mu_lon", "sigma_lat", "sigma_lon"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["mu_lat", "mu_lon"]).copy()
    d["lead_hours"] = d["lead_hours"].fillna(0)
    d = d.sort_values("lead_hours").reset_index(drop=True)
    return d

def load_recovery_predictions(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize recovery prediction file into columns: area_id, t, pred_recovery."""
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    id_col = pick("area_id", "node_id", "cell_id", "fips")
    t_col = pick("t", "time", "step", "time_step")
    v_col = pick("pred_recovery", "recovery", "recovery_index", "yhat")

    if id_col is None or t_col is None or v_col is None:
        return None

    if id_col != "area_id":
        out = out.rename(columns={id_col: "area_id"})
    if t_col != "t":
        out = out.rename(columns={t_col: "t"})
    if v_col != "pred_recovery":
        out = out.rename(columns={v_col: "pred_recovery"})

    out["area_id"] = out["area_id"].astype(str)
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    out["pred_recovery"] = pd.to_numeric(out["pred_recovery"], errors="coerce")
    out = out.dropna(subset=["area_id", "t", "pred_recovery"]).copy()
    out["pred_recovery"] = out["pred_recovery"].clip(0, 1)
    return out



# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def make_track_layers(track: pd.DataFrame, show_uncertainty: bool, highlight_lead: Optional[float] = None) -> List[pdk.Layer]:
    """Build map layers for the path forecast.

    Uses LineLayer segments (more reliable visibility across zoom levels on older pydeck)
    plus point markers for hover tooltips.
    """
    pts = track[["mu_lat", "mu_lon", "lead_hours", "sigma_lat", "sigma_lon"]].copy()
    pts = pts.rename(columns={"mu_lat": "lat", "mu_lon": "lon"})
    pts["title"] = "Path forecast point"
    pts["subtitle"] = pts["lead_hours"].apply(lambda h: f"Lead time: {int(h)} hours")

    # Build line segments between consecutive points
    segs = []
    if len(pts) >= 2:
        coords = pts[["lon", "lat"]].values.tolist()
        for i in range(len(coords) - 1):
            segs.append({"source": coords[i], "target": coords[i+1]})
    segs_df = pd.DataFrame(segs)

    layers: List[pdk.Layer] = []

    if len(segs_df) > 0:
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=segs_df,
                get_source_position="source",
                get_target_position="target",
                get_color=[255, 80, 80],
                get_width=4,
                width_min_pixels=4,
                width_max_pixels=16,
                pickable=False,
            )
        )

    # Base points (for hover + visibility)
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position=["lon", "lat"],
            get_radius=8000,
            radius_min_pixels=8,
            radius_max_pixels=34,
            get_fill_color=[255, 255, 255],
            pickable=True,
        )
    )

    # Highlight a selected time point (if chosen)
    if highlight_lead is not None and len(pts) > 0:
        i = int((pts["lead_hours"] - float(highlight_lead)).abs().idxmin())
        sel = pts.loc[[i]].copy()
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=sel,
                get_position=["lon", "lat"],
                get_radius=14000,
                radius_min_pixels=12,
                radius_max_pixels=40,
                get_fill_color=[255, 200, 0],
                opacity=0.95,
                pickable=True,
            )
        )

    # Uncertainty "bubbles" (optional)
    if show_uncertainty and np.isfinite(pts["sigma_lat"]).any() and np.isfinite(pts["sigma_lon"]).any():
        cone_pts = []
        for _, r in pts.iterrows():
            sig = float(np.nanmean([r["sigma_lat"], r["sigma_lon"]]))
            if not np.isfinite(sig):
                continue
            p50_m = cfg.p50_scale * sig * 111_000
            p90_m = cfg.p90_scale * sig * 111_000
            cone_pts.append({"lat": r["lat"], "lon": r["lon"], "r50": p50_m, "r90": p90_m})

        layers += [
            pdk.Layer(
                "ScatterplotLayer",
                data=cone_pts,
                get_position=["lon", "lat"],
                get_radius="r90",
                opacity=0.14,
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=cone_pts,
                get_position=["lon", "lat"],
                get_radius="r50",
                opacity=0.14,
                pickable=False,
            ),
        ]

    return layers


# Free basemap (no token required). If this URL is blocked, choose 'None' in the sidebar.
FREE_LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
FREE_DARK_STYLE  = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

FREE_LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
FREE_DARK_STYLE  = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

def pydeck_map(
    layers: List[pdk.Layer],
    center: Tuple[float, float],
    zoom: int,
    guide_text: str = "Map guide: hover a dot for details. Hotter colors mean higher need.",
):
    """Render a token-free map using a free public basemap style (no Mapbox key)."""
    view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=0)

    tooltip = {
        "html": (
            "<b>{title}</b><br/>"
            "<span style='font-size:13px'>{subtitle}</span><br/>"
            "<hr style='margin:6px 0'/>"
            f"<span style='font-size:12px'>{guide_text}</span>"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    style_choice = st.session_state.get("map_style_choice", "Light (free)")
    if style_choice == "None":
        map_style = ""
    elif style_choice == "Dark (free)":
        map_style = FREE_DARK_STYLE
    else:
        map_style = FREE_LIGHT_STYLE

    deck = pdk.Deck(
        layers=list(layers),
        initial_view_state=view,
        tooltip=tooltip,
        map_style=map_style,
    )
    st.pydeck_chart(deck, width="stretch")

def heatmap_layer(df: pd.DataFrame, weight_col: str, radius_pixels: int = 70) -> pdk.Layer:
    d = df.copy()
    d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0)
    return pdk.Layer(
        "HeatmapLayer",
        data=d,
        get_position=["lon", "lat"],
        get_weight=weight_col,
        radiusPixels=radius_pixels,
        aggregation="MEAN",
    )

def simple_legend(title: str, left: str, right: str):
    st.markdown(
        f"""
        <div style="margin-top:6px;margin-bottom:4px;">
          <div style="font-size:0.95rem;font-weight:600;">{title}</div>
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:0.85rem;">{left}</div>
            <div style="flex:1;height:10px;border-radius:8px;background:linear-gradient(90deg, rgba(0,0,0,0.08), rgba(0,0,0,0.6));"></div>
            <div style="font-size:0.85rem;">{right}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Plain-language scoring for actions (no technical terms)
# -----------------------------------------------------------------------------
def compute_area_priority(areas: pd.DataFrame, recovery_at_t: Optional[pd.Series]) -> pd.Series:
    """
    Priority = combination of:
      - vulnerability (if available) OR baseline
      - resource scarcity (if nearby facility info exists)
      - slow recovery (1 - recovery_index)
    """
    base = pd.Series(0.5, index=areas.index, dtype=float)

    # vulnerability proxy
    vuln = base.copy()
    if "RPL_THEMES" in areas.columns:
        vuln = pd.to_numeric(areas["RPL_THEMES"], errors="coerce").fillna(areas["RPL_THEMES"].median()).clip(0, 1)

    # scarcity proxy from nearby resources if present
    shelter_cap = pd.to_numeric(areas.get("shelter_capacity_nearby", 0.0), errors="coerce").fillna(0.0)
    hosp_beds = pd.to_numeric(areas.get("hospital_beds_nearby", 0.0), errors="coerce").fillna(0.0)

    shelter_norm = (shelter_cap - shelter_cap.min()) / (shelter_cap.max() - shelter_cap.min() + 1e-9)
    hosp_norm = (hosp_beds - hosp_beds.min()) / (hosp_beds.max() - hosp_beds.min() + 1e-9)
    scarcity = (1.0 - 0.5 * (shelter_norm + hosp_norm)).clip(0, 1)

    # slow recovery term
    slow = pd.Series(0.0, index=areas.index, dtype=float)
    if recovery_at_t is not None:
        slow = (1.0 - recovery_at_t).clip(0, 1)

    priority = (0.45 * vuln + 0.25 * scarcity + 0.30 * slow).clip(0, 1)
    return priority

def recommended_action(priority: float) -> str:
    if priority >= 0.75:
        return "Send support now (shelter + medical + supplies)"
    if priority >= 0.50:
        return "Targeted support (monitor closely + stage resources)"
    return "Monitor (lower urgency)"


# -----------------------------------------------------------------------------
# Comparison helpers (two-spot comparison)
# -----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def nearest_path_info(area_lat, area_lon, track_df: Optional[pd.DataFrame]):
    """Returns (min_distance_km, closest_lead_hours) to the forecast path points."""
    if track_df is None or len(track_df) == 0:
        return (np.nan, np.nan)
    lats = track_df["mu_lat"].to_numpy()
    lons = track_df["mu_lon"].to_numpy()
    d = haversine_km(area_lat, area_lon, lats, lons)
    i = int(np.nanargmin(d))
    return (float(d[i]), float(track_df["lead_hours"].iloc[i]))

def label_risk(x: float) -> str:
    if not np.isfinite(x):
        return "Unknown"
    if x >= 0.75: return "High"
    if x >= 0.50: return "Medium"
    return "Low"

def label_recovery_gap(x: float) -> str:
    if not np.isfinite(x):
        return "Unknown"
    if x >= 0.65: return "Slow recovery likely"
    if x >= 0.35: return "Moderate recovery"
    return "Faster recovery likely"


def generate_comparison_brief(a: dict, b: dict) -> str:
    """Plain-language comparison using Gemini if available (fallback to template)."""
    import os

    def template():
        return f"""
**Two-spot comparison (plain language)**

**Spot A: {a['name']}**
- Storm closeness: {a['storm_km']:.1f} km
- Risk level: {a['risk_label']}
- Recommended action: {a['action']}
- Recovery outlook: {a['recovery_label']}

**Spot B: {b['name']}**
- Storm closeness: {b['storm_km']:.1f} km
- Risk level: {b['risk_label']}
- Recommended action: {b['action']}
- Recovery outlook: {b['recovery_label']}

**Quick takeaway**
- Prioritize the location with higher risk and slower recovery.
- Heatmap reminder: darker/hotter means more need.
"""

    api_key = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_key_input")
    if not api_key:
        return template()

    try:

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.0-pro")

        prompt = f"""
Write a short, non-technical emergency planning comparison.
Avoid technical terms. Write for a first-time viewer.

Spot A:
- Name: {a['name']}
- Distance to storm path (km): {a['storm_km']:.1f}
- Risk level: {a['risk_label']}
- Action: {a['action']}
- Recovery outlook: {a['recovery_label']}

Spot B:
- Name: {b['name']}
- Distance to storm path (km): {b['storm_km']:.1f}
- Risk level: {b['risk_label']}
- Action: {b['action']}
- Recovery outlook: {b['recovery_label']}

End with one clear recommendation.
Also explain in one sentence:
"Darker heatmap color means higher need."
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception:
        return template()
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        prompt = f"""Write a short, non-technical decision brief comparing two places.
Avoid technical terms. Use 6-10 bullet points and one final recommendation line.
Heatmap meaning in one sentence: darker/hotter means more need.

Spot A: {a}
Spot B: {b}
"""
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        return resp.output_text
    except Exception:
        return template()


def gemini_explain(kind: str) -> str:
    """Ask Gemini for a short, non-technical explanation of a map/graph section."""
    import os
    k = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_key_input")
    if not k:
        return "Gemini is not enabled. Set GEMINI_API_KEY to turn on automatic explanations."
    try:

        genai.configure(api_key=k)
        model = genai.GenerativeModel("gemini-1.0-pro")
        prompts = {
            "risk": "Explain this risk heatmap to a non-technical emergency planner in 4-6 bullets. Define what hotter/darker means and how to use it with the storm path line.",
            "recovery": "Explain this recovery heatmap to a non-technical emergency planner in 4-6 bullets. Define what hotter/darker means (slower recovery) and how to use it for resource planning.",
            "path": "Explain this path map to a non-technical viewer in 4-6 bullets. Explain that the line is the expected movement over time, and how to read the hover/time info.",
        }
        prompt = prompts.get(kind, "Explain this dashboard section in simple terms, in 4-6 bullets.")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini explanation failed: {e}"


if _HAS_PLOTLY:
    def plotly_path_map(track: Optional[pd.DataFrame], lead_sel: Optional[float], center: Tuple[float, float], zoom: float):
        """Token-free path visualization using Plotly + OpenStreetMap."""
        fig = go.Figure()

        if track is None or len(track) == 0:
            fig.update_layout(
                height=540,
                margin=dict(l=0, r=0, t=10, b=0),
                annotations=[dict(
                    text="No path data loaded yet.",
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
                )],
            )
            return fig

        tr = track.sort_values("lead_hours").copy()
        # Full path line
        fig.add_trace(go.Scattermapbox(
            lat=tr["mu_lat"],
            lon=tr["mu_lon"],
            mode="lines",
            name="Forecast path",
            hoverinfo="skip",
        ))

        # Path points (hoverable)
        subtitle = tr.apply(
            lambda r: f"Hours ahead: {float(r['lead_hours']):.0f}<br>Lat/Lon: {float(r['mu_lat']):.3f}, {float(r['mu_lon']):.3f}",
            axis=1
        )

        fig.add_trace(go.Scattermapbox(
            lat=tr["mu_lat"],
            lon=tr["mu_lon"],
            mode="markers",
            name="Path points",
            marker=dict(size=8),
            text=subtitle,
            hovertemplate="<b>Forecast point</b><br>%{text}<br><br>"
                          "<span style='font-size:12px'>Guide: the line shows where the storm/flood is expected to move over time.</span><extra></extra>",
        ))

        # Highlight selected lead hour
        if lead_sel is not None and np.isfinite(lead_sel):
            # nearest point
            idx = int((tr["lead_hours"] - float(lead_sel)).abs().idxmin())
            r = tr.loc[idx]
            fig.add_trace(go.Scattermapbox(
                lat=[r["mu_lat"]],
                lon=[r["mu_lon"]],
                mode="markers",
                name="Selected time",
                marker=dict(size=14, symbol="star"),
                hovertemplate="<b>Selected time</b><br>"
                              f"Hours ahead: {float(r['lead_hours']):.0f}<br>"
                              f"Lat/Lon: {float(r['mu_lat']):.3f}, {float(r['mu_lon']):.3f}"
                              "<extra></extra>",
            ))

        fig.update_layout(
            height=540,
            margin=dict(l=0, r=0, t=10, b=0),
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=float(center[0]), lon=float(center[1])),
                zoom=float(zoom),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        )
        return fig

    # -----------------------------------------------------------------------------
    # App
    # -----------------------------------------------------------------------------

else:
    def plotly_path_map(*args, **kwargs):
        return None

def main():
    st.set_page_config(page_title="Unified Decision Dashboard", layout="wide")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        st.selectbox(
            "Basemap style",
            ["Light (free)", "Dark (free)", "None"],
            key="map_style_choice",
            help="Uses a free public basemap (no Mapbox token). Choose 'None' if your network blocks external styles.",
        )

        # Gemini key (session-only; not stored in file)
        gem_key = st.text_input(
            "Gemini API key (optional)",
            type="password",
            help="If set, Gemini can generate plain-language explanations. This stays only in your current session.",
            key="gemini_key_input",
        )


    # ✅ Force light theme (override any earlier styles)
    st.markdown(
        """
        <style>
          .stApp { background-color: #ffffff !important; }
          section[data-testid="stSidebar"] { background-color: #f7f7f9 !important; }
          .stMarkdown, .stCaption, .stText, p, li { color: rgba(0,0,0,0.86) !important; }
          div[data-testid="stDataFrame"] { background-color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Light theme background
    st.markdown("""
    <style>
      .stApp { background-color: #ffffff; }
      section[data-testid="stSidebar"] { background-color: #f7f7f9; }
      .stMarkdown, .stCaption, .stText, p, li { color: rgba(0,0,0,0.86) !important; }
      div[data-testid="stDataFrame"] { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)


    # --- Darker app background (non-white) ---
    st.markdown(
        """
        <style>
          /* Dark background */
          .stApp { background-color: #ffffff; }
          /* Sidebar slightly different */
          section[data-testid="stSidebar"] { background-color: #f7f7f9; }
          /* Make text readable */
          .stMarkdown, .stCaption, .stText, p, li { color: rgba(0,0,0,0.86) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌀🌦️ Decision Dashboard (Demo)")
    st.caption(
        "This dashboard turns forecasts into clear decisions: where the storm may go, which places are most at risk, "
        "what actions to take, and where recovery may be slow."
    )

    # ---------------- Sidebar: uploads + controls ----------------
    st.sidebar.header("📂 Upload data (optional)")
    track_file = st.sidebar.file_uploader("Path forecast (CSV)", type=["csv"])
    shelters_file = st.sidebar.file_uploader("Shelters (CSV)", type=["csv"])
    hospitals_file = st.sidebar.file_uploader("Hospitals (CSV)", type=["csv"])
    schools_file = st.sidebar.file_uploader("Schools (CSV)", type=["csv"])
    areas_file = st.sidebar.file_uploader("Areas grid (CSV)", type=["csv"], help="Areas with lat/lon and area_id (or node_id/cell_id/FIPS)")
    recovery_file = st.sidebar.file_uploader("Recovery predictions (CSV)", type=["csv"], help="area_id + t + pred_recovery in [0,1]")

    st.sidebar.divider()
    show_uncertainty = st.sidebar.toggle("Show uncertainty around the path", value=True)
    zoom = st.sidebar.slider("Map zoom", min_value=3, max_value=10, value=cfg.default_zoom)
    radius_px = st.sidebar.slider("Heatmap spread", min_value=40, max_value=120, value=70, step=10)

    # ---------------- Load data ----------------
    track = normalize_track_df(_try_read_csv(track_file)) if track_file else None
    shelters = normalize_latlon(_try_read_csv(shelters_file)) if shelters_file else None
    hospitals = normalize_latlon(_try_read_csv(hospitals_file)) if hospitals_file else None
    schools = normalize_latlon(_try_read_csv(schools_file)) if schools_file else None

    areas = _try_read_csv(areas_file) if areas_file else None
    areas = ensure_area_id(normalize_latlon(areas)) if areas is not None else None

    if areas is not None and len(areas)>0:
        st.sidebar.caption(f"Areas loaded: {len(areas):,}")
        with st.sidebar.expander('Areas grid preview'):
            st.dataframe(areas.head(20), width='stretch')

    rec = load_recovery_predictions(_try_read_csv(recovery_file)) if recovery_file else None

    # Time slider for recovery (shared across tabs when available)
    t = None
    if rec is not None and len(rec) > 0:
        tmin, tmax = int(rec["t"].min()), int(rec["t"].max())
        t = st.sidebar.slider("Recovery time (t)", tmin, tmax, tmin, 1)

    # ---------------- Tabs ----------------
    tab_path, tab_risk, tab_actions, tab_recovery = st.tabs([
        "🌀 Path forecast",
        "🔥 Risk view",
        "🧭 Recommended actions",
        "🛠 Recovery view",
    ])

    # ---------------- Tab: Path ----------------
    with tab_path:
        st.subheader("Where the storm may go (with uncertainty)")
        st.write(
            "The solid line is the most likely path. The shaded area around it shows uncertainty: wider shading means less certainty."
        )

        if track is None:
            st.info("Upload a path forecast CSV to see the path view.")
        else:
            # Choose a safe slider step (avoid 0 if lead times repeat)
            lead_vals = pd.to_numeric(track["lead_hours"], errors="coerce").dropna().sort_values().values
            if len(lead_vals) >= 2:
                diffs = np.diff(lead_vals)
                diffs = diffs[diffs > 0]
                step_val = float(diffs.min()) if len(diffs) else 1.0
            else:
                step_val = 1.0

            lead_sel = st.slider(
                "Move along the forecast path (hours ahead)",
                float(track["lead_hours"].min()),
                float(track["lead_hours"].max()),
                float(track["lead_hours"].min()),
                step=step_val,
            )
            layers = make_track_layers(track, show_uncertainty=show_uncertainty, highlight_lead=lead_sel)
            center = (float(track["mu_lat"].iloc[0]), float(track["mu_lon"].iloc[0]))
            if _HAS_PLOTLY:
                st.plotly_chart(plotly_path_map(track, lead_sel if 'lead_sel' in locals() else None, center=center, zoom=zoom), width='stretch')
            else:
                st.warning("Plotly is not installed. Install it with: pip install plotly")
                pydeck_map(layers, center=center, zoom=zoom, guide_text="Path map: the line shows where the storm/flood is expected to move over time. Use the slider to step through hours; hover points for the exact time.")

            with st.expander("Explain this path map (Gemini)"):
                st.write(gemini_explain("path"))

            st.divider()
            st.subheader("Compare two places (actions, risk, recovery + storm movement)")
            st.write(
                "Pick two places (areas) to compare. This summary is written in plain language so a new viewer can understand it quickly."
            )

            if areas is None or len(areas) < 2:
                st.info("Upload an Areas grid CSV (with area_id, lat, lon) to enable comparison.")
            else:
                joined = areas.copy()

                # Attach current-time recovery if available
                if rec is not None and t is not None:
                    rec_t = rec[rec["t"] == t][["area_id", "pred_recovery"]].copy()
                    joined = joined.merge(rec_t, on="area_id", how="left")
                    med = float(rec_t["pred_recovery"].median()) if len(rec_t) else 0.5
                    joined["pred_recovery"] = pd.to_numeric(joined["pred_recovery"], errors="coerce").fillna(med).clip(0, 1)
                else:
                    joined["pred_recovery"] = np.nan

                recovery_at_t = joined["pred_recovery"] if joined["pred_recovery"].notna().any() else None
                joined["priority"] = compute_area_priority(joined, recovery_at_t)
                joined["action"] = joined["priority"].apply(recommended_action)
                joined["recovery_gap"] = (1.0 - pd.to_numeric(joined["pred_recovery"], errors="coerce")).clip(0, 1)

                ids = joined["area_id"].astype(str).tolist()
                colA, colB = st.columns(2)
                with colA:
                    a_id = st.selectbox("Spot A", ids, index=0, key="spotA")
                with colB:
                    b_id = st.selectbox("Spot B", ids, index=min(1, len(ids)-1), key="spotB")

                a_row = joined[joined["area_id"].astype(str) == str(a_id)].iloc[0]
                b_row = joined[joined["area_id"].astype(str) == str(b_id)].iloc[0]

                a_km, a_lead = nearest_path_info(float(a_row["lat"]), float(a_row["lon"]), track)
                b_km, b_lead = nearest_path_info(float(b_row["lat"]), float(b_row["lon"]), track)

                a = {
                    "name": str(a_id),
                    "storm_km": float(a_km) if np.isfinite(a_km) else np.nan,
                    "lead_h": float(a_lead) if np.isfinite(a_lead) else np.nan,
                    "risk_score": float(a_row["priority"]),
                    "risk_label": label_risk(float(a_row["priority"])),
                    "action": str(a_row["action"]),
                    "recovery_gap": float(a_row["recovery_gap"]) if np.isfinite(float(a_row["recovery_gap"])) else np.nan,
                    "recovery_label": label_recovery_gap(float(a_row["recovery_gap"])) if np.isfinite(float(a_row["recovery_gap"])) else "Unknown",
                }
                b = {
                    "name": str(b_id),
                    "storm_km": float(b_km) if np.isfinite(b_km) else np.nan,
                    "lead_h": float(b_lead) if np.isfinite(b_lead) else np.nan,
                    "risk_score": float(b_row["priority"]),
                    "risk_label": label_risk(float(b_row["priority"])),
                    "action": str(b_row["action"]),
                    "recovery_gap": float(b_row["recovery_gap"]) if np.isfinite(float(b_row["recovery_gap"])) else np.nan,
                    "recovery_label": label_recovery_gap(float(b_row["recovery_gap"])) if np.isfinite(float(b_row["recovery_gap"])) else "Unknown",
                }

                show = pd.DataFrame({
                    "Metric": [
                        "Storm closeness (km)",
                        "Closest time on path (hours ahead)",
                        "Risk score (0–1)",
                        "Suggested action",
                        "Recovery gap (0–1)",
                    ],
                    "Spot A": [a["storm_km"], a["lead_h"], a["risk_score"], a["action"], a["recovery_gap"]],
                    "Spot B": [b["storm_km"], b["lead_h"], b["risk_score"], b["action"], b["recovery_gap"]],
                })
                st.dataframe(show, width="stretch")

                st.markdown("### Decision summary (auto)")
                st.markdown(generate_comparison_brief(a, b))

                st.divider()
                st.subheader("More map views (risk and recovery)")
                st.caption("These extra views help a first-time viewer quickly see where the need is highest and where recovery may take longer.")

                map_df = joined.copy()
                map_df["risk_score"] = pd.to_numeric(map_df["priority"], errors="coerce").fillna(0.0).clip(0, 1)
                map_df["recovery_gap"] = pd.to_numeric(map_df["recovery_gap"], errors="coerce").fillna(0.0).clip(0, 1)
                map_df["title"] = "Area"
                map_df["subtitle"] = map_df.apply(
                    lambda r: f"Area {r['area_id']} • Risk {r['risk_score']:.2f} • Recovery gap {r['recovery_gap']:.2f}",
                    axis=1,
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Risk map (where urgent action is needed)**")
                    with st.expander("Explain this risk map (Gemini)"):
                        st.write(gemini_explain("risk"))
                    risk_layers = []
                    if track is not None and len(track) > 0:
                        risk_layers += make_track_layers(track, show_uncertainty=False, highlight_lead=None)

                    risk_layers.append(
                        pdk.Layer(
                            "HeatmapLayer",
                            data=map_df,
                            get_position=["lon", "lat"],
                            get_weight="risk_score",
                            radius_pixels=60,
                            opacity=0.55,
                        )
                    )
                    # Hover points (invisible-ish but pickable)
                    risk_layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=map_df,
                            get_position=["lon", "lat"],
                            get_radius=6000,
                            radius_min_pixels=6,
                            radius_max_pixels=20,
                            opacity=0.25,
                            pickable=True,
                        )
                    )
                    pydeck_map(risk_layers, center=center, zoom=zoom, guide_text="Risk map: hotter areas mean higher urgency. Hover a dot to see the area's risk + recovery numbers.")

                with col2:
                    st.markdown("**Recovery map (where support may be needed longer)**")
                    with st.expander("Explain this recovery map (Gemini)"):
                        st.write(gemini_explain("recovery"))
                    rec_layers = []
                    if track is not None and len(track) > 0:
                        rec_layers += make_track_layers(track, show_uncertainty=False, highlight_lead=None)

                    rec_layers.append(
                        pdk.Layer(
                            "HeatmapLayer",
                            data=map_df,
                            get_position=["lon", "lat"],
                            get_weight="recovery_gap",
                            radius_pixels=60,
                            opacity=0.55,
                        )
                    )
                    rec_layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=map_df,
                            get_position=["lon", "lat"],
                            get_radius=6000,
                            radius_min_pixels=6,
                            radius_max_pixels=20,
                            opacity=0.25,
                            pickable=True,
                        )
                    )
                    pydeck_map(rec_layers, center=center, zoom=zoom, guide_text="Recovery map: hotter areas mean slower recovery / longer support needed. Hover a dot to see details.")

                st.divider()
                st.subheader("More graphs (quick comparisons)")

                g1, g2 = st.columns(2)

                with g1:
                    st.markdown("**Risk vs recovery (Spot A vs Spot B)**")
                    bars = pd.DataFrame({
                        "Spot": ["A", "A", "B", "B"],
                        "Metric": ["Risk", "Recovery gap", "Risk", "Recovery gap"],
                        "Value": [a["risk_score"], a["recovery_gap"], b["risk_score"], b["recovery_gap"]],
                    })
                    st.bar_chart(bars, x="Metric", y="Value", color="Spot", width="stretch")

                with g2:
                    st.markdown("**Recovery over time (if available)**")
                    if rec is not None and "t" in rec.columns:
                        rec_a = rec[rec["area_id"].astype(str) == str(a_id)][["t", "pred_recovery"]].copy()
                        rec_b = rec[rec["area_id"].astype(str) == str(b_id)][["t", "pred_recovery"]].copy()
                        rec_a["Spot"] = "A"
                        rec_b["Spot"] = "B"
                        ts = pd.concat([rec_a, rec_b], ignore_index=True)
                        ts["pred_recovery"] = pd.to_numeric(ts["pred_recovery"], errors="coerce").clip(0, 1)
                        st.line_chart(ts, x="t", y="pred_recovery", color="Spot", width="stretch")
                    else:
                        st.info("Upload recovery predictions to see the recovery-over-time curve.")

            with st.expander("Path data preview"):
                st.dataframe(track.head(50), width='stretch')

    # ---------------- Tab: Risk ----------------
    with tab_risk:
        st.subheader("Where impacts could be highest")
        st.write(
            "This heatmap highlights places that could need more attention based on what is nearby "
            "(e.g., hospitals/schools) and whether those places are exposed."
        )

        # Build risk points from facilities if provided
        points = []
        if hospitals is not None and len(hospitals) > 0:
            tmp = hospitals.copy()
            beds_raw = tmp["beds"] if "beds" in tmp.columns else pd.Series(100, index=tmp.index)
            beds = pd.to_numeric(beds_raw, errors="coerce").fillna(100)
            tmp["risk"] = (1.0 / np.sqrt(beds)).clip(0, 1)  # fewer beds => higher concern
            tmp["type"] = "Hospital"
            points.append(tmp[["lat", "lon", "risk", "type"]])

        if schools is not None and len(schools) > 0:
            tmp = schools.copy()
            kids_raw = tmp["children_est"] if "children_est" in tmp.columns else (tmp["capacity"] if "capacity" in tmp.columns else pd.Series(300, index=tmp.index))
            kids = pd.to_numeric(kids_raw, errors="coerce").fillna(300)
            tmp["risk"] = np.clip(kids / 1500.0, 0, 1)  # more children => higher concern
            tmp["type"] = "School"
            points.append(tmp[["lat", "lon", "risk", "type"]])

        if len(points) == 0:
            st.info("Upload hospitals and/or schools CSV to see a risk heatmap in this tab.")
        else:
            risk_df = pd.concat(points, ignore_index=True)
            layers = [heatmap_layer(risk_df, "risk", radius_pixels=radius_px)]
            center = (float(risk_df["lat"].mean()), float(risk_df["lon"].mean()))
            pydeck_map(layers, center=center, zoom=zoom)

            simple_legend("Heatmap meaning", "Lower concern", "Higher concern")
            st.caption("Tip: The heatmap is a summary view; hover/click points in other tabs for details.")

            with st.expander("Risk points preview"):
                st.dataframe(risk_df.head(50), width='stretch')

    # ---------------- Tab: Actions ----------------
    with tab_actions:
        st.subheader("What to do next (auto-ranked)")
        st.write(
            "These suggestions are generated from the maps above. "
            "They are written in plain language so that anyone can act on them quickly."
        )

        if areas is None:
            st.info("Upload an Areas grid CSV to generate ranked actions.")
        else:
            # Recovery at time t (optional)
            recovery_at_t = None
            if rec is not None and t is not None:
                rec_t = rec[rec["t"] == t][["area_id", "pred_recovery"]].copy()
                joined = areas.merge(rec_t, on="area_id", how="left")
                # fill missing with median
                med = float(rec_t["pred_recovery"].median()) if len(rec_t) else 0.5
                recovery_at_t = pd.to_numeric(joined["pred_recovery"], errors="coerce").fillna(med).clip(0, 1)
            else:
                joined = areas.copy()

            priority = compute_area_priority(joined, recovery_at_t)
            joined["priority"] = priority
            joined["action"] = joined["priority"].apply(recommended_action)

            top = joined[["area_id", "priority", "action"]].sort_values("priority", ascending=False).head(25)
            st.dataframe(top, width='stretch')

            st.caption(
                "How to read this table: higher priority means the area likely needs attention sooner "
                "(higher vulnerability, fewer nearby resources, and/or slower predicted recovery)."
            )

    # ---------------- Tab: Recovery ----------------
    with tab_recovery:
        st.subheader("Recovery heatmap (where recovery may be slow)")
        st.write(
            "This view shows **where recovery may be slower**, so response teams can prioritize support. "
            "Hotter areas on the heatmap mean a larger recovery gap (more help likely needed)."
        )

        if areas is None or rec is None or t is None:
            st.info("Upload Areas grid CSV and Recovery predictions CSV to see the recovery heatmap.")
        else:
            rec_t = rec[rec["t"] == t][["area_id", "pred_recovery"]].copy()
            joined = areas.merge(rec_t, on="area_id", how="left")

            # If missing values exist, fill with median to keep map stable
            med = float(rec_t["pred_recovery"].median()) if len(rec_t) else 0.5
            joined["pred_recovery"] = pd.to_numeric(joined["pred_recovery"], errors="coerce").fillna(med).clip(0, 1)

            # We plot "recovery gap" so hotspots = slower recovery
            joined["recovery_gap"] = (1.0 - joined["pred_recovery"]).clip(0, 1)

            layers = [heatmap_layer(joined[["lat", "lon", "recovery_gap"]], "recovery_gap", radius_pixels=radius_px)]
            center = (float(joined["lat"].mean()), float(joined["lon"].mean()))
            pydeck_map(layers, center=center, zoom=zoom)

            # Plain-language legend
            simple_legend("Heatmap meaning", "Faster recovery", "Slower recovery (needs support)")

            # Optional: show top slow-recovery areas
            st.markdown("**Top areas needing support (at this time step):**")
            worst = joined[["area_id", "pred_recovery", "recovery_gap"]].sort_values("recovery_gap", ascending=False).head(15)
            worst = worst.rename(columns={"pred_recovery": "recovery_index (0–1)", "recovery_gap": "recovery_gap (0–1)"})
            st.dataframe(worst, width='stretch')

            st.caption(
                "Recovery index: 1 means quicker/better recovery. "
                "Recovery gap: 1 means a larger gap and likely more help needed."
            )

if __name__ == "__main__":
    main()
