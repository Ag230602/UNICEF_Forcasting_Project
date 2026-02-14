# unicef_another_updated.py
# Streamlit + PyDeck dashboard (STORM-CARE) with:
# - 4 operational map panels: Rescue / Resources / Population / Risk
# - Dynamic hurricane path overlay (live NHC if available + scenario simulator fallback)
# - County-driven scenario selection (impact focus auto-updates by selected county)
# - Gemini (LLM) explainability + optional "Speech Assistant" (text-to-speech-ready script)
#
# Live data sources:
# - NHC CurrentStorms.json (active storms metadata + product links)
# - NOAA/NHC ArcGIS MapServer (forecast/past track geometries)
#
# Docs:
# - NHC CurrentStorms JSON: https://www.nhc.noaa.gov/CurrentStorms.json
# - NOAA NHC Tropical Weather MapServer:
#   https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer
#
# Install:
#   pip install streamlit pydeck pandas numpy requests pyshp shapely google-generativeai
#
# Run:
#   python -m streamlit run unicef_another_updated.py --server.port 8501

import base64
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import shapefile  # pyshp
import streamlit as st
from shapely.geometry import Point, shape

# =========================
# 🔑 GEMINI API KEY (INLINE)
# =========================
# NOTE: Avoid committing keys to GitHub. Prefer Streamlit secrets or environment variable.
GEMINI_API_KEY_INLINE = os.getenv("GEMINI_API_KEY_INLINE", "").strip()

# =========================
# SAFE GEMINI IMPORT
# =========================
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_LIB_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_LIB_AVAILABLE = False

# =========================
# YOUR LOCAL FACILITY FILES
# =========================
# Tip: for portability, use relative paths inside your repo. These are kept as-is from your original file.
SCHOOLS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\schools.csv")
HOSPITALS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\hospitals.csv")
SHELTERS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\shelters.csv")

# =========================
# AUTO-DOWNLOAD PUBLIC DATA
# =========================
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CB_COUNTY_2023_ZIP = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
ACS_YEAR = 2023
ACS_PROFILE_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile"
STATE_FIPS_FL = "12"

DP05_TOTAL = "DP05_0001E"  # total population
DP05_U18 = "DP05_0019E"    # under 18

# =========================
# NOAA / NHC live storm feeds
# =========================
NHC_CURRENT_STORMS_JSON = "https://www.nhc.noaa.gov/CurrentStorms.json"
NHC_MAPSERVER_BASE = "https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer"

# UNICEF-style palette (discrete bins)
UNICEF_BINS = [
    (0, 20,  [241, 248, 255, 210]),
    (20, 40, [198, 230, 255, 210]),
    (40, 60, [126, 198, 255, 210]),
    (60, 80, [ 66, 160, 255, 210]),
    (80, 100,[ 18, 110, 220, 220]),
]


def score_to_bin_color(score: float):
    s = float(np.clip(score, 0, 100))
    for lo, hi, rgba in UNICEF_BINS:
        if lo <= s < hi or (hi == 100 and s <= 100):
            return rgba
    return UNICEF_BINS[0][2]


# =========================
# I/O helpers
# =========================
def download_if_missing(url: str, out_path: Path):
    if out_path.exists():
        return out_path
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def extract_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        return out_dir
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")
    return out_dir


def shp_to_geojson_counties_fl(shp_dir: Path):
    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {shp_dir}")
    shp_path = shp_files[0]

    r = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in r.fields[1:]]

    features = []
    for sr in r.shapeRecords():
        rec = dict(zip(fields, sr.record))
        if str(rec.get("STATEFP", "")).zfill(2) != STATE_FIPS_FL:
            continue

        geom = sr.shape.__geo_interface__
        geoid = rec.get("GEOID")
        if not geoid:
            geoid = f"{str(rec.get('STATEFP')).zfill(2)}{str(rec.get('COUNTYFP')).zfill(3)}"
        rec["GEOID"] = str(geoid)

        features.append({
            "type": "Feature",
            "properties": rec,
            "geometry": geom,
        })

    return {"type": "FeatureCollection", "features": features}


def load_facilities(path: Path, kind: str):
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "kind"])
    df = pd.read_csv(path)
    if "X" in df.columns and "Y" in df.columns:
        df = df.rename(columns={"X": "lon", "Y": "lat"})
    if "lon" not in df.columns or "lat" not in df.columns:
        return pd.DataFrame(columns=["lon", "lat", "kind"])
    df = df[["lon", "lat"]].copy()
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])]
    df["kind"] = kind
    return df


def fetch_acs_dp05_counties_fl(census_key: str | None):
    params = {
        "get": f"NAME,{DP05_TOTAL},{DP05_U18}",
        "for": "county:*",
        "in": f"state:{STATE_FIPS_FL}",
    }
    if census_key:
        params["key"] = census_key

    r = requests.get(ACS_PROFILE_URL, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"ACS API failed: {r.status_code} {r.text[:300]}")
    data = r.json()
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["GEOID"] = df["state"] + df["county"]

    df[DP05_TOTAL] = pd.to_numeric(df[DP05_TOTAL], errors="coerce").fillna(0.0)
    df[DP05_U18] = pd.to_numeric(df[DP05_U18], errors="coerce").fillna(0.0)

    df["child_share_pct"] = np.where(df[DP05_TOTAL] > 0, 100.0 * df[DP05_U18] / df[DP05_TOTAL], 0.0)
    return df[["GEOID", "NAME", DP05_TOTAL, DP05_U18, "child_share_pct"]].copy()


def assign_points_to_counties(points_df: pd.DataFrame, counties_geojson: dict):
    if points_df.empty:
        return points_df

    polys = []
    props = []
    for feat in counties_geojson["features"]:
        geom = shape(feat["geometry"])
        polys.append(geom)
        props.append(feat["properties"])

    bboxes = [p.bounds for p in polys]

    county_geoid = []
    for lon, lat in zip(points_df["lon"].to_numpy(), points_df["lat"].to_numpy()):
        pt = Point(float(lon), float(lat))
        found = None
        for (minx, miny, maxx, maxy), poly, pr in zip(bboxes, polys, props):
            if not (minx <= pt.x <= maxx and miny <= pt.y <= maxy):
                continue
            if poly.contains(pt):
                found = pr["GEOID"]
                break
        county_geoid.append(found)

    out = points_df.copy()
    out["GEOID"] = county_geoid
    return out.dropna(subset=["GEOID"]).copy()


def compute_support_resource_score(acs_df, schools_pts, hospitals_pts, shelters_pts):
    def counts(df):
        return df.groupby("GEOID").size().rename("count")

    base = acs_df.set_index("GEOID").copy()
    base["schools_n"] = counts(schools_pts).reindex(base.index).fillna(0).astype(float) if not schools_pts.empty else 0.0
    base["hospitals_n"] = counts(hospitals_pts).reindex(base.index).fillna(0).astype(float) if not hospitals_pts.empty else 0.0
    base["shelters_n"] = counts(shelters_pts).reindex(base.index).fillna(0).astype(float) if not shelters_pts.empty else 0.0

    denom = np.maximum(base[DP05_U18].to_numpy(), 1.0)
    base["schools_per_10k_children"] = 10000.0 * base["schools_n"] / denom
    base["hospitals_per_10k_children"] = 10000.0 * base["hospitals_n"] / denom
    base["shelters_per_10k_children"] = 10000.0 * base["shelters_n"] / denom

    raw = (
        0.45 * base["hospitals_per_10k_children"] +
        0.35 * base["shelters_per_10k_children"] +
        0.20 * base["schools_per_10k_children"]
    )

    rmin, rmax = float(raw.min()), float(raw.max())
    score = np.zeros(len(raw), dtype=float) if (rmax - rmin) < 1e-9 else (100.0 * (raw - rmin) / (rmax - rmin))

    base["support_resource_score"] = score
    base["pediatric_vulnerability"] = base["child_share_pct"]
    base["population_risk"] = base[DP05_U18]
    return base.reset_index()


def attach_scores_to_geojson(counties_geojson: dict, scores_df: pd.DataFrame, choropleth_opacity: float):
    lookup = scores_df.set_index("GEOID").to_dict(orient="index")
    for feat in counties_geojson["features"]:
        gid = str(feat["properties"].get("GEOID", "")).strip()
        rec = lookup.get(gid, None)
        if rec is None:
            feat["properties"]["support_resource_score"] = None
            feat["properties"]["pediatric_vulnerability"] = None
            feat["properties"]["population_risk"] = None
            feat["properties"]["fill_color"] = [240, 240, 240, int(255 * choropleth_opacity)]
            continue

        s = float(rec.get("support_resource_score", 0.0))
        pv = float(rec.get("pediatric_vulnerability", 0.0))
        pr = float(rec.get("population_risk", 0.0))
        nm = rec.get("NAME", "")

        rgba = score_to_bin_color(s)
        rgba = [rgba[0], rgba[1], rgba[2], int(rgba[3] * choropleth_opacity)]

        feat["properties"]["support_resource_score"] = round(s, 2)
        feat["properties"]["pediatric_vulnerability"] = round(pv, 2)
        feat["properties"]["population_risk"] = round(pr, 0)
        feat["properties"]["ACS_NAME"] = nm
        feat["properties"]["fill_color"] = rgba

    return counties_geojson


def centroids_from_geojson(counties_geojson: dict, scores_df: pd.DataFrame):
    lookup = scores_df.set_index("GEOID").to_dict(orient="index")
    pts = []
    for feat in counties_geojson["features"]:
        gid = str(feat["properties"].get("GEOID", "")).strip()
        rec = lookup.get(gid)
        if not rec:
            continue
        geom = shape(feat["geometry"])
        c = geom.centroid
        pts.append({"lon": float(c.x), "lat": float(c.y), "weight": float(rec.get("population_risk", 0.0)), "GEOID": gid})
    return pd.DataFrame(pts)


def county_centroid_lookup(counties_geojson: dict) -> Dict[str, Tuple[float, float, str]]:
    out = {}
    for feat in counties_geojson["features"]:
        gid = str(feat["properties"].get("GEOID", "")).strip()
        nm = str(feat["properties"].get("NAME", "")).strip()
        geom = shape(feat["geometry"])
        c = geom.centroid
        out[gid] = (float(c.x), float(c.y), nm)
    return out


# =========================
# Gemini (LLM) helpers
# =========================
def get_gemini_key() -> Optional[str]:
    if GEMINI_API_KEY_INLINE:
        return GEMINI_API_KEY_INLINE
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
            return key if key else None
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


@st.cache_data(show_spinner=False)
def gemini_cached(prompt: str, model_name: str) -> str:
    key = get_gemini_key()
    if (not key) or (not GEMINI_LIB_AVAILABLE):
        return ""
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = getattr(resp, "text", "") or ""
        return txt.strip()
    except Exception:
        return ""


def llm_or_fallback(prompt: str, model_name: str, fallback: str) -> str:
    out = gemini_cached(prompt, model_name=model_name)
    return out if out.strip() else fallback


# =========================
# Hurricane path: live + scenario
# =========================
@dataclass
class TrackPoint:
    lon: float
    lat: float
    t_hr: float
    wind_kt: Optional[float] = None
    label: str = ""


def _arcgis_query(layer_id: int, where: str = "1=1", out_fields: str = "*") -> dict:
    url = f"{NHC_MAPSERVER_BASE}/{layer_id}/query"
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "pjson",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def list_active_storm_wallets() -> List[Dict[str, str]]:
    """
    Uses CurrentStorms.json to return a list of active storms (basin+id) + friendly label.
    If no storms, returns empty list.
    """
    try:
        r = requests.get(NHC_CURRENT_STORMS_JSON, timeout=30)
        if r.status_code != 200:
            return []
        js = r.json()
        # Format: keys are storm IDs like "al012020" with nested properties
        storms = []
        for storm_id, rec in js.items():
            if not isinstance(rec, dict):
                continue
            name = str(rec.get("stormName", "")).strip() or storm_id
            basin = str(rec.get("basin", "")).strip()
            number = str(rec.get("stormNumber", "")).strip()
            label = f"{name} ({storm_id})"
            storms.append({"storm_id": storm_id, "label": label, "basin": basin, "number": number})
        # stable ordering: name
        storms.sort(key=lambda x: x["label"].lower())
        return storms
    except Exception:
        return []


def guess_wallet_from_storm_id(storm_id: str) -> Optional[str]:
    """
    Map CurrentStorms.json storm_id (e.g., al012026) to MapServer 'wallet' group (AT1/AT2/EP1/...).
    This mapping is not directly provided in the JSON, so we do a best-effort guess:
    - Atlantic: AT{n}
    - East Pacific: EP{n}
    - Central Pacific: CP{n}
    """
    sid = storm_id.lower().strip()
    if len(sid) < 4:
        return None
    basin = sid[:2]
    num = sid[2:4]
    try:
        n = int(num)
    except Exception:
        return None
    if basin == "al":
        return f"AT{n}"
    if basin == "ep":
        return f"EP{n}"
    if basin == "cp":
        return f"CP{n}"
    return None


def load_live_track(wallet: str) -> Tuple[List[TrackPoint], List[TrackPoint]]:
    """
    Pull forecast points and past points from MapServer layers.
    Wallet example: AT1, AT2, EP1 ...
    Layer ids from service page:
      Forecast Points: 6 (AT1), 32 (AT2), ...
      Forecast Track:  7 (AT1), 33 (AT2), ...
      Past Points:     11 (AT1), 37 (AT2), ...
      Past Track:      12 (AT1), 38 (AT2), ...
    Because layer IDs shift per wallet, we query the service listing once and build the ids.
    """
    # Cache the layer dictionary from the service "All layers" endpoint
    svc = requests.get(f"{NHC_MAPSERVER_BASE}?f=pjson", timeout=30).json()
    layers = svc.get("layers", [])
    # Map layer name -> id
    name_to_id = {str(l.get("name")): int(l.get("id")) for l in layers if "name" in l and "id" in l}

    fc_points_name = f"{wallet} Forecast Points"
    past_points_name = f"{wallet} Past Points"

    fc_id = name_to_id.get(fc_points_name)
    past_id = name_to_id.get(past_points_name)

    def _pts(layer_id: Optional[int]) -> List[TrackPoint]:
        if layer_id is None:
            return []
        js = _arcgis_query(layer_id=layer_id)
        feats = js.get("features", []) or []
        pts: List[TrackPoint] = []
        for f in feats:
            geom = (f.get("geometry") or {})
            attrs = (f.get("attributes") or {})
            lon = float(geom.get("x"))
            lat = float(geom.get("y"))
            # Best effort: different fields exist depending on storm/product
            t_hr = float(attrs.get("TAU", attrs.get("tau", attrs.get("FcstHr", 0.0))) or 0.0)
            wind = attrs.get("MAXWIND", attrs.get("maxwind", None))
            wind_kt = float(wind) if wind is not None and str(wind).strip() != "" else None
            label = str(attrs.get("STORMNAME", attrs.get("stormname", "")) or "")
            pts.append(TrackPoint(lon=lon, lat=lat, t_hr=t_hr, wind_kt=wind_kt, label=label))
        pts.sort(key=lambda p: p.t_hr)
        return pts

    return _pts(past_id), _pts(fc_id)


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    lam1, phi1, lam2, phi2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlam = lam2 - lam1
    dphi = phi2 - phi1
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2.0)**2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def bearing_deg(lon1, lat1, lon2, lat2) -> float:
    lam1, phi1, lam2, phi2 = map(np.radians, [lon1, lat1, lon2, lat2])
    y = np.sin(lam2 - lam1) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(lam2 - lam1)
    brng = np.degrees(np.arctan2(y, x))
    return float((brng + 360.0) % 360.0)


def simulate_track(
    start_lon: float,
    start_lat: float,
    heading_deg0: float,
    speed_kmh: float,
    turn_rate_deg_per_hr: float,
    total_hours: int,
    step_hr: int,
    jitter_km: float,
    seed: int,
) -> List[TrackPoint]:
    rng = np.random.default_rng(seed)
    pts = [TrackPoint(lon=float(start_lon), lat=float(start_lat), t_hr=0.0, label="SIM")]
    heading = float(heading_deg0)
    # Very simple geographic step (small deltas), sufficient for a *demo overlay*
    for t in range(step_hr, total_hours + step_hr, step_hr):
        heading = (heading + turn_rate_deg_per_hr * step_hr) % 360.0
        dist_km = max(0.0, speed_kmh * step_hr)
        # jitter to emulate scenario uncertainty
        dist_km = max(0.0, dist_km + rng.normal(0, jitter_km))
        # convert km to degrees (approx)
        dlat = (dist_km / 111.0) * np.cos(np.radians(heading))
        dlon = (dist_km / (111.0 * np.cos(np.radians(pts[-1].lat)) + 1e-9)) * np.sin(np.radians(heading))
        pts.append(TrackPoint(lon=float(pts[-1].lon + dlon), lat=float(pts[-1].lat + dlat), t_hr=float(t), label="SIM"))
    return pts


def scenario_params_from_llm(
    county_name: str,
    scenario_text: str,
    model_name: str,
) -> Dict[str, float]:
    """
    Ask the LLM for scenario parameters in strict JSON.
    If unavailable, return a reasonable default.
    """
    fallback = {
        "heading_deg0": 340.0,
        "speed_kmh": 22.0,
        "turn_rate_deg_per_hr": 0.25,
        "jitter_km": 6.0,
    }
    prompt = (
        "You are helping simulate a hurricane forecast track for a demo dashboard.\n"
        "Return ONLY a JSON object with numeric keys:\n"
        "heading_deg0 (0-360), speed_kmh (5-60), turn_rate_deg_per_hr (-2 to 2), jitter_km (0-30).\n"
        "Base it on the scenario text AND the target Florida county, keeping values plausible.\n\n"
        f"Target county: {county_name}\n"
        f"Scenario text: {scenario_text}\n"
    )
    txt = gemini_cached(prompt, model_name=model_name)
    if not txt:
        return fallback
    try:
        js = json.loads(txt.strip().splitlines()[-1])
        out = {}
        for k in fallback:
            out[k] = float(js.get(k, fallback[k]))
        # clamp
        out["heading_deg0"] = float(np.clip(out["heading_deg0"], 0, 360))
        out["speed_kmh"] = float(np.clip(out["speed_kmh"], 5, 60))
        out["turn_rate_deg_per_hr"] = float(np.clip(out["turn_rate_deg_per_hr"], -2, 2))
        out["jitter_km"] = float(np.clip(out["jitter_km"], 0, 30))
        return out
    except Exception:
        return fallback


# =========================
# Arrow icon (inline SVG as data URL)
# =========================
_ARROW_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<polygon points="32,2 60,62 32,52 4,62" fill="#111"/>
</svg>"""
ARROW_ICON_URL = "data:image/svg+xml;base64," + base64.b64encode(_ARROW_SVG.encode("utf-8")).decode("ascii")


def build_arrow_layer(arrow_rows: pd.DataFrame, size: int = 32) -> pdk.Layer:
    return pdk.Layer(
        "IconLayer",
        data=arrow_rows,
        get_icon="icon_data",
        get_position="[lon, lat]",
        get_size=size,
        size_scale=1,
        get_angle="angle",
        pickable=False,
    )


def build_track_layers(track_points: List[TrackPoint], show_points: bool = True) -> List[pdk.Layer]:
    if not track_points or len(track_points) < 2:
        return []

    df = pd.DataFrame([{"lon": p.lon, "lat": p.lat, "t_hr": p.t_hr} for p in track_points])
    # PathLayer expects a list of paths; one path here
    path_df = pd.DataFrame([{"path": df[["lon", "lat"]].values.tolist()}])

    # Arrow at "current" segment (last segment)
    lon1, lat1 = float(df.iloc[-2]["lon"]), float(df.iloc[-2]["lat"])
    lon2, lat2 = float(df.iloc[-1]["lon"]), float(df.iloc[-1]["lat"])
    ang = bearing_deg(lon1, lat1, lon2, lat2)
    arrow_rows = pd.DataFrame([{
        "lon": lon2,
        "lat": lat2,
        "angle": ang,
        "icon_data": {
            "url": ARROW_ICON_URL,
            "width": 64,
            "height": 64,
            "anchorY": 32,
            "anchorX": 32,
        },
    }])

    layers: List[pdk.Layer] = [
        pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            width_scale=20,
            width_min_pixels=3,
            get_color=[255, 255, 255],
            opacity=0.95,
            pickable=False,
        ),
        build_arrow_layer(arrow_rows, size=26),
    ]

    if show_points:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_radius=16000,
                radius_min_pixels=4,
                get_fill_color=[255, 255, 255],
                pickable=False,
                opacity=0.85,
            )
        )
    return layers


# =========================
# Explainability (view + speech assistant)
# =========================
def build_view_context(scores_f: pd.DataFrame, selected_geoid: str, scenario_mode: str, storm_label: str, extra: dict):
    sel_row = None
    if selected_geoid and len(scores_f):
        s = scores_f[scores_f["GEOID"] == selected_geoid]
        if len(s):
            sel_row = s.iloc[0].to_dict()

    ctx = {
        "selected_county_geoid": selected_geoid or None,
        "selected_county": (sel_row or {}).get("NAME") if sel_row else None,
        "scenario_mode": scenario_mode,
        "storm_label": storm_label,
        "summary_metrics": {
            "counties_in_view": int(len(scores_f)),
            "avg_support_score": float(scores_f["support_resource_score"].mean()) if len(scores_f) else None,
            "total_children_u18_in_view": float(scores_f[DP05_U18].sum()) if len(scores_f) else 0.0,
        },
        "selected_county_metrics": {
            "support_resource_score": float((sel_row or {}).get("support_resource_score")) if sel_row else None,
            "child_share_pct": float((sel_row or {}).get("child_share_pct")) if sel_row else None,
            "children_u18": float((sel_row or {}).get(DP05_U18)) if sel_row else None,
            "hospitals_n": float((sel_row or {}).get("hospitals_n")) if sel_row else None,
            "shelters_n": float((sel_row or {}).get("shelters_n")) if sel_row else None,
            "schools_n": float((sel_row or {}).get("schools_n")) if sel_row else None,
        },
        "track_diagnostics": extra,
        "known_limitations": [
            "Forecast track overlay is for situational awareness and demo visualization; always consult official NHC products for operations.",
            "Facility layers depend on your local CSVs; if missing, those points will not appear.",
            "Support Resource Score is a proxy (weighted facilities per 10k children), min-max scaled within Florida.",
        ],
    }
    return ctx


def fallback_explain(ctx: dict) -> str:
    lines = []
    lines.append(f"Selected county: {ctx.get('selected_county') or 'None'}")
    lines.append(f"Scenario mode: {ctx.get('scenario_mode')} | Storm: {ctx.get('storm_label') or 'N/A'}")
    sm = ctx.get("summary_metrics", {})
    lines.append(f"Counties in view: {sm.get('counties_in_view')}, total children (U18) in view: {sm.get('total_children_u18_in_view'):.0f}")
    sel = ctx.get("selected_county_metrics", {})
    if sel.get("support_resource_score") is not None:
        lines.append(
            f"Selected county score: {sel.get('support_resource_score'):.1f} "
            f"(child share {sel.get('child_share_pct'):.1f}%, U18 {sel.get('children_u18'):.0f})"
        )
    td = ctx.get("track_diagnostics", {}) or {}
    if td:
        lines.append(f"Track: min distance to selected county ≈ {td.get('min_dist_km', 'n/a')} km; ETA ≈ {td.get('eta_hr', 'n/a')} hours.")
    return "\n".join(lines)


def llm_explain(ctx: dict, model_name: str) -> str:
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    prompt = (
        "You are a UNICEF-style emergency dashboard narrator.\n"
        "Explain what the user is seeing, with emphasis on children, facilities, and the storm track.\n"
        "Use ONLY the JSON context.\n"
        "If something is missing, say it is missing.\n"
        "Return 6-10 bullet points, concise, operational tone.\n\n"
        f"Context JSON:\n{ctx_json}\n"
    )
    txt = gemini_cached(prompt, model_name=model_name)
    return txt if txt else fallback_explain(ctx)


def llm_speech_script(ctx: dict, model_name: str) -> str:
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    prompt = (
        "Write a short voice narration (45-70 seconds) for a live demo.\n"
        "Audience: UNICEF-style decision makers.\n"
        "Must mention: selected county, child population, support score, resource gaps, and storm track ETA/distance.\n"
        "Avoid numbers you cannot see in context.\n\n"
        f"Context JSON:\n{ctx_json}\n"
    )
    txt = gemini_cached(prompt, model_name=model_name)
    if txt:
        return txt
    # fallback
    base = fallback_explain(ctx)
    return "Demo narration:\n" + base.replace("\n", ". ") + "."


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="STORM-CARE (Dynamic Track + County Scenarios)", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color:#666; font-size: 12px; }
.kpi { background: #ffffff; border: 1px solid #e7e7e7; border-radius: 14px; padding: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("STORM-CARE — Dynamic Hurricane Path + County Scenarios")
st.caption("4 panels: Rescue / Resources / Population / Risk — each with a live/simulated storm track overlay and an LLM-based narration option.")

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Primary child lens")
    pv_min = st.slider("Min child share (%)", 0, 60, 20)
    child_min = st.slider("Min child population (U18)", 0, 500000, 5000, step=1000)

    st.divider()
    st.header("Map styles")
    zoom = st.slider("Zoom", 4.0, 10.5, 6.3, 0.1)
    use_dark = st.checkbox("Use dark basemap (contrast)", False)

    st.divider()
    st.header("Layers")
    show_choro = st.checkbox("Show Support Resource Score (county)", True)
    choro_opacity = st.slider("Choropleth opacity", 0.20, 1.00, 0.88, 0.02)
    line_width = st.slider("County boundary thickness", 1, 5, 2)

    hm_radius = st.slider("Heatmap radius (px)", 40, 220, 120, 5)
    hm_intensity = st.slider("Heatmap intensity", 0.5, 3.0, 1.2, 0.1)

    st.divider()
    st.header("Census API (optional)")
    census_key = st.text_input("Census API Key", value="", type="password")

    st.divider()
    st.header("Storm path mode")
    scenario_mode = st.radio("Track source", ["Live (NHC)", "Scenario (AI-assisted)", "Scenario (Manual)"], index=1)

    # Live storms
    live_storm_label = "None"
    live_storm_id = ""
    if scenario_mode == "Live (NHC)":
        storms = list_active_storm_wallets()
        if storms:
            sel = st.selectbox("Active storm", storms, format_func=lambda x: x["label"])
            live_storm_label = sel["label"]
            live_storm_id = sel["storm_id"]
        else:
            st.info("No active storms found right now (NHC feed). Switch to Scenario mode for demo.")

    # Scenario knobs
    scenario_text = st.text_area(
        "Scenario description",
        value="A major hurricane approaches Florida from the Caribbean; track bends slightly west as it nears landfall.",
        height=90,
    )
    sim_total_hours = st.slider("Forecast horizon (hours)", 24, 144, 120, 6)
    sim_step_hr = st.slider("Step (hours)", 3, 12, 6, 1)
    sim_seed = st.number_input("Simulation seed", value=7, min_value=1, max_value=9999)

    if scenario_mode == "Scenario (Manual)":
        manual_heading = st.slider("Heading (deg)", 0, 360, 340)
        manual_speed = st.slider("Speed (km/h)", 5, 60, 22)
        manual_turn = st.slider("Turn rate (deg/hr)", -2.0, 2.0, 0.25, 0.05)
        manual_jitter = st.slider("Uncertainty jitter (km)", 0.0, 30.0, 6.0, 0.5)

    st.divider()
    st.header("LLM (Gemini) + Speech assistant")
    gemini_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    enable_speech = st.toggle("Generate speech narration script", value=False)
    auto_explain = st.toggle("Auto-generate explainability bullets", value=True)

    key_present = bool(get_gemini_key())
    if key_present and GEMINI_LIB_AVAILABLE:
        st.caption("✅ Gemini key + library detected.")
    elif key_present and (not GEMINI_LIB_AVAILABLE):
        st.caption("⚠️ Gemini key detected but library missing. Install: pip install google-generativeai")
    else:
        st.caption("⚠️ No Gemini key found. LLM features will use fallback text.")

# =========================
# LOAD + BUILD (cached)
# =========================
@st.cache_data(show_spinner=True)
def build_all(census_key_: str):
    zip_path = download_if_missing(CB_COUNTY_2023_ZIP, CACHE_DIR / "cb_2023_us_county_500k.zip")
    shp_dir = extract_zip(zip_path, CACHE_DIR / "cb_2023_us_county_500k")
    counties_geo = shp_to_geojson_counties_fl(shp_dir)

    schools = load_facilities(SCHOOLS_CSV, "schools")
    hospitals = load_facilities(HOSPITALS_CSV, "hospitals")
    shelters = load_facilities(SHELTERS_CSV, "shelters")

    schools_c = assign_points_to_counties(schools, counties_geo)
    hospitals_c = assign_points_to_counties(hospitals, counties_geo)
    shelters_c = assign_points_to_counties(shelters, counties_geo)

    acs = fetch_acs_dp05_counties_fl(census_key_.strip() or None)
    scores = compute_support_resource_score(acs, schools_c, hospitals_c, shelters_c)
    return counties_geo, scores, schools, hospitals, shelters


try:
    counties_geo, scores_df, schools, hospitals, shelters = build_all(census_key)
except Exception as e:
    st.error(f"Failed to build base data: {e}")
    st.stop()

# Apply pediatric filter
scores_f = scores_df[
    (scores_df["pediatric_vulnerability"] >= pv_min) &
    (scores_df[DP05_U18] >= child_min)
].copy()

# Defensive: deep copy geojson before property edits
counties_geo2 = json.loads(json.dumps(counties_geo))
counties_geo2 = attach_scores_to_geojson(counties_geo2, scores_f, choropleth_opacity=choro_opacity)
pop_pts = centroids_from_geojson(counties_geo2, scores_f)

centroids = county_centroid_lookup(counties_geo2)

# County selector (scenario focus)
all_county_choices = [{"GEOID": r["GEOID"], "NAME": r["NAME"]} for _, r in scores_df.sort_values("NAME").iterrows()]
selected = st.selectbox("Focus county (scenario-based view)", all_county_choices, format_func=lambda x: x["NAME"])
selected_geoid = selected["GEOID"]
sel_lon, sel_lat, sel_nm = centroids.get(selected_geoid, (-82.2, 27.8, selected["NAME"]))

# =========================
# Build hurricane track (live or scenario)
# =========================
storm_label = ""
past_pts: List[TrackPoint] = []
fc_pts: List[TrackPoint] = []

if scenario_mode == "Live (NHC)" and live_storm_id:
    wallet = guess_wallet_from_storm_id(live_storm_id)
    storm_label = live_storm_label
    if wallet:
        try:
            past_pts, fc_pts = load_live_track(wallet)
        except Exception:
            past_pts, fc_pts = [], []
else:
    storm_label = "Scenario track"
    # Start point: place south of Florida, adjust slightly toward focused county
    start_lon, start_lat = -81.2, 22.6
    if scenario_mode == "Scenario (AI-assisted)":
        params = scenario_params_from_llm(county_name=sel_nm, scenario_text=scenario_text, model_name=gemini_model)
    else:
        params = {
            "heading_deg0": float(manual_heading),
            "speed_kmh": float(manual_speed),
            "turn_rate_deg_per_hr": float(manual_turn),
            "jitter_km": float(manual_jitter),
        }
    # Nudge heading toward county (soft constraint)
    toward = bearing_deg(start_lon, start_lat, sel_lon, sel_lat)
    params["heading_deg0"] = float((0.7 * params["heading_deg0"] + 0.3 * toward) % 360.0)

    fc_pts = simulate_track(
        start_lon=start_lon,
        start_lat=start_lat,
        heading_deg0=params["heading_deg0"],
        speed_kmh=params["speed_kmh"],
        turn_rate_deg_per_hr=params["turn_rate_deg_per_hr"],
        total_hours=int(sim_total_hours),
        step_hr=int(sim_step_hr),
        jitter_km=params["jitter_km"],
        seed=int(sim_seed),
    )
    past_pts = []

# Combine for diagnostics & overlays
track_for_diag = (past_pts + fc_pts) if past_pts else fc_pts
min_dist = None
eta_hr = None
if track_for_diag:
    dists = [haversine_km(p.lon, p.lat, sel_lon, sel_lat) for p in track_for_diag]
    min_dist = float(np.min(dists))
    # ETA: closest point time
    idx = int(np.argmin(dists))
    eta_hr = float(track_for_diag[idx].t_hr)

track_diag = {
    "min_dist_km": round(min_dist, 1) if min_dist is not None else None,
    "eta_hr": round(eta_hr, 1) if eta_hr is not None else None,
    "track_points_n": int(len(track_for_diag)) if track_for_diag else 0,
}

# =========================
# KPIs
# =========================
shown_counties = int(len(scores_f))
avg_score = float(scores_f["support_resource_score"].mean()) if shown_counties else 0.0
median_child_share = float(scores_f["pediatric_vulnerability"].median()) if shown_counties else 0.0
total_children = float(scores_f[DP05_U18].sum()) if shown_counties else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f'<div class="kpi"><div class="small-muted">Counties shown</div><div style="font-size:22px;font-weight:800;">{shown_counties}</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="small-muted">Avg Support Score</div><div style="font-size:22px;font-weight:800;">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="small-muted">Median Child Share (%)</div><div style="font-size:22px;font-weight:800;">{median_child_share:.1f}</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="small-muted">Children in view (U18)</div><div style="font-size:22px;font-weight:800;">{total_children:,.0f}</div></div>', unsafe_allow_html=True)
k5.markdown(
    f'<div class="kpi"><div class="small-muted">Track min dist / ETA</div><div style="font-size:18px;font-weight:800;">{track_diag.get("min_dist_km") or "—"} km / {track_diag.get("eta_hr") or "—"} hr</div></div>',
    unsafe_allow_html=True
)

# =========================
# Shared base layers (counties)
# =========================
def county_layer() -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer",
        data=counties_geo2,
        pickable=True,
        opacity=1.0,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=line_width,
    )


def tooltip_html(title: str) -> dict:
    return {
        "html": f"""
        <div style="font-family: Arial; font-size: 12px; line-height: 1.35;">
          <div style="font-size: 13px; font-weight: 800;">{title}</div>
          <div style="margin-top: 6px;">
            <div><b>County:</b> {{NAME}}</div>
            <div><b>Support Resource Score:</b> {{support_resource_score}}</div>
            <div><b>Child share (%):</b> {{pediatric_vulnerability}}</div>
            <div><b>Children (U18):</b> {{population_risk}}</div>
          </div>
        </div>
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }


def focused_view() -> pdk.ViewState:
    return pdk.ViewState(latitude=float(sel_lat), longitude=float(sel_lon), zoom=float(zoom), pitch=0)


def build_deck(layers: List[pdk.Layer], tooltip: dict) -> pdk.Deck:
    map_style = "dark" if use_dark else "light"
    return pdk.Deck(
        layers=layers,
        initial_view_state=focused_view(),
        map_style=map_style,
        tooltip=tooltip,
    )


# =========================
# Track overlay layers (forecast + past)
# =========================
track_layers: List[pdk.Layer] = []
if past_pts:
    track_layers += build_track_layers(past_pts, show_points=False)
if fc_pts:
    track_layers += build_track_layers(fc_pts, show_points=True)

# Marker for selected county
sel_df = pd.DataFrame([{"lon": sel_lon, "lat": sel_lat, "name": sel_nm}])
sel_marker = pdk.Layer(
    "ScatterplotLayer",
    data=sel_df,
    get_position="[lon, lat]",
    get_radius=22000,
    radius_min_pixels=6,
    get_fill_color=[255, 0, 0],
    pickable=False,
    opacity=0.9,
)

# =========================
# 4 PANEL DASHBOARD
# =========================
tabs = st.tabs(["🚑 Rescue map", "🏥 Resources map", "👶 Population map", "⚠️ Risk map", "📋 Table", "🧾 Explain + Speech"])

# --- Rescue map (shelters + track + focused county) ---
with tabs[0]:
    st.caption("Rescue view: shelters density + storm path + focused county. (Use this to plan evacuation + shelter staging.)")
    layers = []
    if show_choro:
        layers.append(county_layer())
    if not shelters.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=shelters,
                get_position="[lon, lat]",
                get_weight=1,
                radius_pixels=hm_radius,
                intensity=hm_intensity,
                threshold=0.05,
            )
        )
    layers += track_layers + [sel_marker]
    st.pydeck_chart(build_deck(layers, tooltip_html("🚑 Rescue view")), use_container_width=True)

# --- Resources map (hospitals + schools + track) ---
with tabs[1]:
    st.caption("Resources view: hospitals + schools density + storm path (for medical + school-as-shelter planning).")
    layers = []
    if show_choro:
        layers.append(county_layer())
    if not hospitals.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=hospitals,
                get_position="[lon, lat]",
                get_weight=1,
                radius_pixels=hm_radius,
                intensity=hm_intensity,
                threshold=0.05,
            )
        )
    if not schools.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=schools,
                get_position="[lon, lat]",
                get_weight=1,
                radius_pixels=max(40, int(hm_radius * 0.85)),
                intensity=hm_intensity,
                threshold=0.05,
            )
        )
    layers += track_layers + [sel_marker]
    st.pydeck_chart(build_deck(layers, tooltip_html("🏥 Resources view")), use_container_width=True)

# --- Population map (children heatmap + track) ---
with tabs[2]:
    st.caption("Population view: child population (U18) heatmap + storm path (where children are concentrated).")
    layers = []
    if show_choro:
        layers.append(county_layer())
    if not pop_pts.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=pop_pts,
                get_position="[lon, lat]",
                get_weight="weight",
                radius_pixels=min(220, int(hm_radius * 1.15)),
                intensity=hm_intensity,
                threshold=0.05,
            )
        )
    layers += track_layers + [sel_marker]
    st.pydeck_chart(build_deck(layers, tooltip_html("👶 Population view")), use_container_width=True)

# --- Risk map (support score + track + quick priority list) ---
with tabs[3]:
    st.caption("Risk view: support score choropleth + storm path. Lower score + high children = priority.")
    layers = []
    if show_choro:
        layers.append(county_layer())
    layers += track_layers + [sel_marker]
    st.pydeck_chart(build_deck(layers, tooltip_html("⚠️ Risk view")), use_container_width=True)

    # quick ranked table for the selected county neighborhood (top 10 lowest score within filter)
    st.markdown("#### Priority shortlist (lowest Support Resource Score among filtered counties)")
    show_cols = ["NAME", "GEOID", "support_resource_score", "child_share_pct", DP05_U18, "hospitals_n", "shelters_n", "schools_n"]
    shortlist = scores_f[show_cols].sort_values("support_resource_score", ascending=True).head(10)
    st.dataframe(shortlist, use_container_width=True, height=320)

# --- Table ---
with tabs[4]:
    st.caption("Auditable evidence table (filtered).")
    show_cols = [
        "GEOID", "NAME", DP05_U18, "child_share_pct", "support_resource_score",
        "schools_n", "hospitals_n", "shelters_n",
        "schools_per_10k_children", "hospitals_per_10k_children", "shelters_per_10k_children",
    ]
    view = scores_f[show_cols].sort_values("support_resource_score", ascending=False).copy()
    st.dataframe(view, use_container_width=True, height=420)

    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered table (CSV)",
        data=csv_bytes,
        file_name="support_resource_score_filtered.csv",
        mime="text/csv",
    )

# --- Explain + Speech ---
with tabs[5]:
    ctx = build_view_context(scores_f=scores_f, selected_geoid=selected_geoid, scenario_mode=scenario_mode, storm_label=storm_label, extra=track_diag)

    if auto_explain:
        st.markdown("### Explainability bullets")
        st.write(llm_explain(ctx, model_name=gemini_model))

    if enable_speech:
        st.markdown("### Speech assistant (demo narration)")
        speech_text = llm_speech_script(ctx, model_name=gemini_model)
        st.text_area("Narration script (copy into any TTS / voice tool)", value=speech_text, height=200)

    with st.expander("Audit context JSON"):
        st.code(json.dumps(ctx, indent=2), language="json")

st.caption("Tip: If Live (NHC) is empty (off-season), switch to Scenario mode for a fully dynamic demo track.")
