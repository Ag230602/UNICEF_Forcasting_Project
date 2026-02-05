# storm_care_app_gemini_ULTIMATE.py
# Streamlit + PyDeck dashboard with:
# 1) AUTO demo mode (safe: never blank due to immediate rerun)
# 2) Gemini explainability narrator (LLM) with fallback if key missing OR library missing
# 3) INLINE Gemini key supported (optional) + secrets/env supported
#
# Install:
#   pip install streamlit pydeck pandas numpy requests pyshp shapely google-generativeai
#
# Gemini key options (highest priority first):
#   1) Inline in this file: GEMINI_API_KEY_INLINE = "..."
#   2) Streamlit secrets: .streamlit/secrets.toml  -> GEMINI_API_KEY="..."
#   3) Environment variable: GEMINI_API_KEY
#
# Run:
#   python -m streamlit run storm_care_app_gemini_ULTIMATE.py --server.port 8501 --logger.level=info

import json
import os
import time
import zipfile
from pathlib import Path

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
# Put your key here if you want the code to be fully standalone.
# If you leave it "", the app will try Streamlit secrets then environment variable.
GEMINI_API_KEY_INLINE = "AIzaSyA0onsn4xElTjgZUP01aByhxbRVy1bPvBU"  # <-- PASTE KEY HERE (recommended to NOT commit to GitHub)


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
# UNICEF-style palette (discrete bins)
# =========================
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


# =========================
# Gemini Explainability
# =========================
def get_gemini_key() -> str | None:
    if GEMINI_API_KEY_INLINE.strip():
        return GEMINI_API_KEY_INLINE.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
            return key if key else None
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


def build_view_context(scores_f: pd.DataFrame, pv_min, child_min, hm_pop, hm_hosp, hm_sch, hm_shel, show_choro):
    ctx = {
        "filters": {
            "min_child_share_pct": int(pv_min),
            "min_child_population_u18": int(child_min),
            "layers": {
                "choropleth_support_score": bool(show_choro),
                "heatmap_children_population": bool(hm_pop),
                "heatmap_hospitals": bool(hm_hosp),
                "heatmap_schools": bool(hm_sch),
                "heatmap_shelters": bool(hm_shel),
            },
        },
        "data_availability": {
            "counties_shown": int(len(scores_f)),
            "avg_support_score": float(scores_f["support_resource_score"].mean()) if len(scores_f) else None,
            "min_support_score": float(scores_f["support_resource_score"].min()) if len(scores_f) else None,
            "max_support_score": float(scores_f["support_resource_score"].max()) if len(scores_f) else None,
            "median_child_share": float(scores_f["pediatric_vulnerability"].median()) if len(scores_f) else None,
            "total_children_u18_in_scope": float(scores_f[DP05_U18].sum()) if len(scores_f) else 0.0,
        },
        "known_limitations": [
            "Support Resource Score is a proxy: weighted facilities per 10k children, then min-max scaled to 0–100 within the filtered set.",
            "Heatmaps depend on local facility CSVs; if missing/empty, that layer will show nothing.",
            "Facility-to-county uses point-in-polygon containment; borderline points may not match.",
            "This is descriptive (not causal).",
        ],
    }
    return ctx


def fallback_explain(ctx: dict, user_question: str = "") -> str:
    f = ctx["filters"]
    d = ctx["data_availability"]
    layers = f["layers"]

    lines = []
    lines.append(f"Counties shown: {d['counties_shown']}.")
    lines.append(f"Filters: child share ≥ {f['min_child_share_pct']}% AND children (U18) ≥ {f['min_child_population_u18']}.")
    if d["counties_shown"] == 0:
        lines.append("No counties match the current filters. Try lowering thresholds.")
        return "\n".join(lines)

    lines.append(f"Support Resource Score range: {d['min_support_score']:.1f}–{d['max_support_score']:.1f} (avg {d['avg_support_score']:.1f}).")
    lines.append("Score meaning: weighted hospitals/shelters/schools per 10k children, scaled 0–100.")
    on_layers = [k for k, v in layers.items() if v]
    lines.append(f"Active layers: {', '.join(on_layers)}.")
    if user_question.strip():
        lines.append("")
        lines.append(f"Your question: {user_question.strip()}")
        lines.append("Note: This view is proxy-based; for operational readiness add staffing/beds/accessibility datasets.")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def gemini_explain_cached(ctx_json: str, user_question: str, model_name: str) -> str:
    ctx = json.loads(ctx_json)
    key = get_gemini_key()

    if (not key) or (not GEMINI_LIB_AVAILABLE):
        return fallback_explain(ctx, user_question)

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)

        prompt = (
            "You are an explainability narrator for a child-focused hurricane support dashboard.\n"
            "Explain what the user is seeing and why, using ONLY the JSON context below.\n"
            "Rules:\n"
            "- Do NOT invent data not present in the context.\n"
            "- If something is missing/empty, explain likely reasons grounded in the context.\n"
            "- Be concise; bullet points are OK.\n"
            "- If the user asks a question, answer it using the context.\n\n"
            f"Context JSON:\n{ctx_json}\n\n"
            f"User question (optional): {user_question.strip()}\n"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text.strip() else fallback_explain(ctx, user_question)
    except Exception:
        return fallback_explain(ctx, user_question)


def llm_explain(ctx: dict, user_question: str, model_name: str) -> str:
    ctx_json = json.dumps(ctx, ensure_ascii=False, sort_keys=True)
    return gemini_explain_cached(ctx_json=ctx_json, user_question=user_question.strip(), model_name=model_name)


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="STORM-CARE (Storm-focused Child-centered Actionable Risk Engine)", layout="wide")

# Heartbeat: if you see these, the app is NOT blank/crashed
st.write("✅ App loaded")
st.sidebar.write("✅ Sidebar loaded")

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

st.title("STORM-CARE (Storm-focused Child-centered Actionable Risk Engine)")
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

    # -------- Demo Mode ----------
    st.divider()
    st.header("Demo mode (auto-run)")
    demo_on = st.toggle("▶️ Run guided demo automatically", value=False)
    demo_speed = st.slider("Demo speed (seconds/step)", 2, 12, 5)

    if "demo_step" not in st.session_state:
        st.session_state.demo_step = 0
    if "demo_last_tick" not in st.session_state:
        st.session_state.demo_last_tick = 0.0
    if "demo_note" not in st.session_state:
        st.session_state.demo_note = ""

    # -------- Gemini Explainer ----------
    st.divider()
    st.header("Explainability (Gemini)")
    explain_auto = st.toggle("Auto-explain when view changes", value=True)
    user_question = st.text_input("Ask about this view (optional)", value="")
    gemini_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

    key_present = bool(get_gemini_key())
    lib_present = GEMINI_LIB_AVAILABLE
    if key_present and lib_present:
        st.caption("✅ Gemini key + library detected.")
    elif key_present and (not lib_present):
        st.caption("⚠️ Gemini key detected but library missing. Install: pip install google-generativeai")
    else:
        st.caption("⚠️ No Gemini key found. Explainer will use fallback text.")


# =========================
# Demo script: safe auto-advance
# =========================
DEMO_STEPS = [
    (10,  2000,  True, True,  False, False, True, 6.2, False, "Start broad: score + hospitals + child population risk."),
    (20,  5000,  True, True,  False, False, True, 6.4, False, "Focus pediatric-first: higher child-share counties."),
    (25, 15000,  True, True,  False, True,  True, 6.6, False, "Add shelters: child risk clusters without shelter density."),
    (30, 25000,  True, True,  True,  True,  True, 6.8, True,  "Dark map + schools: low score + high child risk."),
    (20,  5000,  False, True, True,  False, True, 6.3, False, "Capacity layers only: hospitals+schools vs score."),
]


def apply_demo_step(i: int):
    i = int(i) % len(DEMO_STEPS)
    pv, cm, pop, hosp, sch, shel, choro, zm, dark, note = DEMO_STEPS[i]
    st.session_state.demo_step = i
    st.session_state.demo_note = note

    st.session_state.pv_min_demo = pv
    st.session_state.child_min_demo = cm
    st.session_state.hm_pop_demo = pop
    st.session_state.hm_hosp_demo = hosp
    st.session_state.hm_sch_demo = sch
    st.session_state.hm_shel_demo = shel
    st.session_state.show_choro_demo = choro
    st.session_state.zoom_demo = zm
    st.session_state.use_dark_demo = dark


# SAFE: never rerun immediately on first toggle
if demo_on:
    if st.session_state.demo_last_tick == 0.0:
        st.session_state.demo_last_tick = time.time()
        apply_demo_step(st.session_state.demo_step)
    else:
        now = time.time()
        if now - st.session_state.demo_last_tick >= float(demo_speed):
            st.session_state.demo_last_tick = now
            apply_demo_step(st.session_state.demo_step + 1)
            st.rerun()
else:
    st.session_state.demo_last_tick = 0.0

# Override widget values while demo is running
if demo_on:
    pv_min = int(st.session_state.get("pv_min_demo", pv_min))
    child_min = int(st.session_state.get("child_min_demo", child_min))
    hm_pop = bool(st.session_state.get("hm_pop_demo", hm_pop))
    hm_hosp = bool(st.session_state.get("hm_hosp_demo", hm_hosp))
    hm_sch = bool(st.session_state.get("hm_sch_demo", hm_sch))
    hm_shel = bool(st.session_state.get("hm_shel_demo", hm_shel))
    show_choro = bool(st.session_state.get("show_choro_demo", show_choro))
    zoom = float(st.session_state.get("zoom_demo", zoom))
    use_dark = bool(st.session_state.get("use_dark_demo", use_dark))


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
# Explainability trigger
# =========================
ctx = build_view_context(scores_f, pv_min, child_min, hm_pop, hm_hosp, hm_sch, hm_shel, show_choro)
sig = json.dumps(ctx["filters"], sort_keys=True)

if "last_explain_sig" not in st.session_state:
    st.session_state.last_explain_sig = None
if "last_user_q" not in st.session_state:
    st.session_state.last_user_q = ""
if "explain_text" not in st.session_state:
    st.session_state.explain_text = ""

should_explain = False
if explain_auto and (st.session_state.last_explain_sig != sig):
    should_explain = True
if user_question.strip() and (st.session_state.last_user_q != user_question.strip()):
    should_explain = True

if should_explain:
    st.session_state.explain_text = llm_explain(ctx, user_question=user_question, model_name=gemini_model)
    st.session_state.last_explain_sig = sig
    st.session_state.last_user_q = user_question.strip()


# =========================
# KPI + tabs
# =========================
total_counties = len({f["properties"]["GEOID"] for f in counties_geo2["features"]})
shown_counties = len(scores_f)

avg_score = float(scores_f["support_resource_score"].mean()) if shown_counties else 0.0
median_child_share = float(scores_f["pediatric_vulnerability"].median()) if shown_counties else 0.0
total_children = float(scores_f[DP05_U18].sum()) if shown_counties else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="kpi"><div class="small-muted">Counties shown</div><div style="font-size:24px;font-weight:800;">{shown_counties}/{total_counties}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><div class="small-muted">Avg Support Score</div><div style="font-size:24px;font-weight:800;">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><div class="small-muted">Median Child Share (%)</div><div style="font-size:24px;font-weight:800;">{median_child_share:.1f}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"><div class="small-muted">Children in scope (U18)</div><div style="font-size:24px;font-weight:800;">{total_children:,.0f}</div></div>', unsafe_allow_html=True)

tab_map, tab_table, tab_method = st.tabs(["🗺️ Map", "📋 Table", "🧾 Method"])


# =========================
# MAP TAB
# =========================
with tab_map:
    if demo_on:
        st.info(f"🎬 Demo step {st.session_state.demo_step + 1}/{len(DEMO_STEPS)} — {st.session_state.get('demo_note','')}")
    else:
        st.caption("Tip: Turn on **Demo mode** in the sidebar to see a guided walkthrough automatically.")

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
    st.caption("Exportable and auditable evidence table for the filtered view.")
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


# =========================
# METHOD TAB (includes Gemini explainer)
# =========================
with tab_method:
    st.markdown("### Gemini explainability — why you are seeing this")
    st.write(st.session_state.explain_text or "No explanation yet — adjust filters or enable auto-explain.")

    with st.expander("Audit context (JSON used for explanation)"):
        st.code(json.dumps(ctx, indent=2), language="json")

    st.markdown(
        """
### What the map is showing (transparent + auditable)

- **Base layer (choropleth): Support Resource Score (0–100)** at the **county** level in Florida.  
- **Primary child lens:** counties are filtered by:
  - minimum **child share (%)** and
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
- Discrete bins (not confusing gradients)  
- Transparent metric definition + downloadable evidence table
        """
    )
