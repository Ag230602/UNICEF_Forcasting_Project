import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import pydeck as pdk
import shapefile  # pyshp
from shapely.geometry import shape, Point
import streamlit as st


# =========================
# YOUR LOCAL FACILITY FILES
# =========================
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

DP05_TOTAL = "DP05_0001E"   # total population
DP05_U18 = "DP05_0019E"     # under 18


# =========================
# UNICEF-style palette (discrete bins)
# =========================
# Light -> deep UNICEF blue, + an alert red for lowest bin if desired (optional)
UNICEF_BINS = [
    (0, 20,  [241, 248, 255, 210]),
    (20, 40, [198, 230, 255, 210]),
    (40, 60, [126, 198, 255, 210]),
    (60, 80, [ 66, 160, 255, 210]),
    (80, 100, [ 18, 110, 220, 220]),
]

def score_to_bin_color(score):
    s = float(np.clip(score, 0, 100))
    for lo, hi, rgba in UNICEF_BINS:
        if lo <= s < hi or (hi == 100 and s <= 100):
            return rgba
    return UNICEF_BINS[0][2]


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


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
    base["pediatric_vulnerability"] = base["child_share_pct"]             # explainable proxy
    base["population_risk"] = base[DP05_U18]                              # children count as risk weight proxy
    return base.reset_index()


def attach_scores_to_geojson(counties_geojson: dict, scores_df: pd.DataFrame,
                             choropleth_opacity: float):
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


def make_legend_html():
    items = []
    for lo, hi, rgba in UNICEF_BINS:
        swatch = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]/255:.2f})"
        items.append(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
              <div style="width:18px; height:12px; background:{swatch}; border:1px solid #999;"></div>
              <div style="font-size:12px; color:#222;">{lo}–{hi}</div>
            </div>
            """
        )
    return f"""
    <div style="background:#ffffffcc; border:1px solid #ddd; padding:10px; border-radius:10px;">
      <div style="font-weight:700; font-size:13px; margin-bottom:8px;">Support Resource Score</div>
      {''.join(items)}
      <div style="font-size:11px; color:#555; margin-top:6px;">
        Higher = more child-normalized capacity (hospitals/shelters/schools).
      </div>
    </div>
    """


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="UNICEF-style Decision Map", layout="wide")

st.markdown(
    """
<style>
/* Cleaner dashboard look */
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color:#666; font-size: 12px; }
.kpi { background: #ffffff; border: 1px solid #e7e7e7; border-radius: 14px; padding: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("UNICEF-style Decision Map ")
st.caption("Choropleth Support Resource Score + toggleable heatmaps + pediatric-first filter (reproducible public data + your facilities).")


# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Primary child lens")
    pv_min = st.slider("Min child share (%)", 0, 60, 20)
    child_min = st.slider("Min child population (U18)", 0, 500000, 5000, step=1000)

    st.divider()
    st.header("Choropleth")
    show_choro = st.checkbox("Show Support Resource Score", True)
    choro_opacity = st.slider("Choropleth opacity", 0.20, 1.00, 0.88, 0.02)
    line_width = st.slider("Boundary thickness", 1, 5, 2)

    st.divider()
    st.header("Heatmaps (toggles)")
    hm_pop = st.checkbox("Population risk (children)", True)
    hm_hosp = st.checkbox("Hospitals density", True)
    hm_sch = st.checkbox("Schools density", False)
    hm_shel = st.checkbox("Shelters density", False)

    st.caption("Adjust radius/intensity so layers remain readable.")
    hm_radius = st.slider("Heatmap radius (px)", 40, 220, 120, 5)
    hm_intensity = st.slider("Heatmap intensity", 0.5, 3.0, 1.2, 0.1)

    st.divider()
    st.header("Map")
    zoom = st.slider("Zoom", 4.0, 9.5, 6.2, 0.1)
    use_dark = st.checkbox("Use dark basemap (strong contrast)", False)

    st.divider()
    st.header("Census API (optional)")
    census_key = st.text_input("Census API Key", value="", type="password")
    st.caption("No key works for light usage; key recommended for repeated runs.")


# =========================
# LOAD + BUILD (cached)
# =========================
@st.cache_data(show_spinner=True)
def build_all(census_key: str):
    zip_path = download_if_missing(CB_COUNTY_2023_ZIP, CACHE_DIR / "cb_2023_us_county_500k.zip")
    shp_dir = extract_zip(zip_path, CACHE_DIR / "cb_2023_us_county_500k")
    counties_geo = shp_to_geojson_counties_fl(shp_dir)

    schools = load_facilities(SCHOOLS_CSV, "schools")
    hospitals = load_facilities(HOSPITALS_CSV, "hospitals")
    shelters = load_facilities(SHELTERS_CSV, "shelters")

    schools_c = assign_points_to_counties(schools, counties_geo)
    hospitals_c = assign_points_to_counties(hospitals, counties_geo)
    shelters_c = assign_points_to_counties(shelters, counties_geo)

    acs = fetch_acs_dp05_counties_fl(census_key.strip() or None)
    scores = compute_support_resource_score(acs, schools_c, hospitals_c, shelters_c)
    return counties_geo, scores, schools, hospitals, shelters


try:
    counties_geo, scores_df, schools, hospitals, shelters = build_all(census_key)
except Exception as e:
    st.error(f"Failed to build map: {e}")
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


# =========================
# KPI + tabs
# =========================
total_counties = len({f["properties"]["GEOID"] for f in counties_geo2["features"]})
shown_counties = len(scores_f)

# Simple KPIs 
avg_score = float(scores_f["support_resource_score"].mean()) if shown_counties else 0.0
median_child_share = float(scores_f["pediatric_vulnerability"].median()) if shown_counties else 0.0
total_children = float(scores_f[DP05_U18].sum()) if shown_counties else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="kpi"><div class="small-muted">Counties shown</div><div style="font-size:24px;font-weight:800;">{shown_counties}/{total_counties}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><div class="small-muted">Avg Support Score</div><div style="font-size:24px;font-weight:800;">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><div class="small-muted">Median Child Share (%)</div><div style="font-size:24px;font-weight:800;">{median_child_share:.1f}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"><div class="small-muted">Children in scope (U18)</div><div style="font-size:24px;font-weight:800;">{total_children:,.0f}</div></div>', unsafe_allow_html=True)

tab_map, tab_table, tab_method = st.tabs(["🗺️ Map", "📋 Table", "🧾 Method "])


# =========================
# MAP TAB
# =========================
with tab_map:
    # Legend shown above map (nice and clean)
    # st.markdown(make_legend_html(), unsafe_allow_html=True)

    layers = []

    if show_choro:
        layers.append(
            pdk.Layer(
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
        )

    # Heatmaps (facility points)
    if hm_hosp and not hospitals.empty:
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
    if hm_sch and not schools.empty:
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
    if hm_shel and not shelters.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=shelters,
                get_position="[lon, lat]",
                get_weight=1,
                radius_pixels=max(40, int(hm_radius * 0.85)),
                intensity=hm_intensity,
                threshold=0.05,
            )
        )

    # Population risk (county centroids weighted by children count)
    if hm_pop and not pop_pts.empty:
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

    view_state = pdk.ViewState(latitude=27.8, longitude=-82.2, zoom=zoom, pitch=0)

    tooltip = {
        "html": """
        <div style="font-family: Arial; font-size: 12px; line-height: 1.35;">
          <div style="font-size: 13px; font-weight: 800;">{NAME}</div>
          <div style="margin-top: 6px;">
            <div><b>Support Resource Score:</b> {support_resource_score}</div>
            <div><b>Child share (%):</b> {pediatric_vulnerability}</div>
            <div><b>Children (U18):</b> {population_risk}</div>
          </div>
          <div style="margin-top: 8px; opacity: 0.75;">
            Capacity proxy = hospitals/shelters/schools per 10k children (weighted).
          </div>
        </div>
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    map_style = "dark" if use_dark else "light"

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)


# =========================
# TABLE TAB
# =========================
with tab_table:
    st.caption("Exportable and auditable. This is typically asks for to justify visuals.")
    show_cols = ["GEOID", "NAME", DP05_U18, "child_share_pct", "support_resource_score",
                 "schools_n", "hospitals_n", "shelters_n",
                 "schools_per_10k_children", "hospitals_per_10k_children", "shelters_per_10k_children"]
    view = scores_f[show_cols].sort_values("support_resource_score", ascending=False).copy()
    st.dataframe(view, use_container_width=True, height=420)

    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered table (CSV)", data=csv_bytes, file_name="support_resource_score_filtered.csv", mime="text/csv")


# =========================
# METHOD TAB
# =========================
with tab_method:
    st.markdown(
        """
### What the map is showing (style — transparent and auditable)

- **Base layer (choropleth): Support Resource Score (0–100)** at the **county** level in Florida.  
- **Primary child lens:** counties are filtered by:
  - minimum **child share (%)** and/or
  - minimum **child population (Under 18)**  
- **Overlays (toggleable heatmaps):**
  - **Hospitals density**
  - **Schools density**
  - **Shelters density**
  - **Population risk density** (weighted by county **child population**)

### How the Support Resource Score is computed (explainable proxy)
1) Count facilities inside each county (spatial join from your point datasets).  
2) Normalize by children: compute facilities per **10,000 children**.  
3) Weighted capacity proxy:
- 0.45 × hospitals per 10k children  
- 0.35 × shelters per 10k children  
- 0.20 × schools per 10k children  
4) Min–max scale to **0–100** for interpretability and UNICEF-style communication.

### Why this meets UNICEF visualization expectations
- Aggregated, policy-relevant regional map  
- Child-first filtering and risk emphasis  
- Clean discrete choropleth legend (no confusing gradients)  
- Transparent metric definition + downloadable evidence table
        """
    )
