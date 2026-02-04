import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =====================================================
# BASE FOLDER (ONLY CHANGE THIS IF NEEDED)
# =====================================================
BASE_DIR = r"C:\Users\Adrija\Downloads\DFGCN"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_OUT = os.path.join(OUT_DIR, "fig1_track_evolution.png")
FIG2_OUT = os.path.join(OUT_DIR, "fig2_recovery_context.png")

MAP_EXTENT = (-180, 180, -90, 90)

SHOW_ARROWS = True
SHOW_TIME_LABELS = True
TIME_LABEL_EVERY = 1
DX, DY = 2.5, 1.5


# =====================================================
# AUTO-FIND HELPERS
# =====================================================
def find_file_anywhere(root, filename_candidates):
    """
    Search recursively under root for any of the candidate filenames.
    Returns the newest match if multiple exist.
    """
    matches = []
    for name in filename_candidates:
        matches.extend(glob.glob(os.path.join(root, "**", name), recursive=True))

    if not matches:
        raise FileNotFoundError(
            "Could not find any of these files under:\n"
            f"{root}\n\n"
            f"Searched for: {filename_candidates}"
        )

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]

def pick_background_image(folder):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    if not files:
        raise FileNotFoundError(
            f"No map image found in: {folder}\n"
            f"Put a world map image there (png/jpg)."
        )
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def find_first_col(df, candidates, required=True, name="df"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"[{name}] missing one of: {candidates}\nAvailable: {list(df.columns)}")
    return None

def ensure_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def add_arrows(ax, lon, lat, color, lw=2):
    for i in range(len(lon) - 1):
        ax.annotate(
            "",
            xy=(lon[i + 1], lat[i + 1]),
            xytext=(lon[i], lat[i]),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color),
            zorder=6
        )

def add_time_labels(ax, lon, lat, lead_hours, color):
    for i in range(0, len(lead_hours), TIME_LABEL_EVERY):
        ax.text(
            lon[i] + DX,
            lat[i] + DY,
            f"{int(lead_hours[i])}h",
            fontsize=9,
            weight="bold",
            color=color,
            zorder=7
        )


# =====================================================
# FIND INPUT FILES AUTOMATICALLY
# =====================================================
PRED_CSV  = find_file_anywhere(BASE_DIR, [
    "inference_test_predictions_all_models.csv",
    "*predictions*all*models*.csv",
    "*predictions*.csv"
])

REC_CSV   = find_file_anywhere(BASE_DIR, [
    "pred_recovery.csv",
    "*recovery*.csv"
])

NODES_CSV = find_file_anywhere(BASE_DIR, [
    "nodes.csv",
    "*nodes*.csv"
])

MAP_IMG   = pick_background_image(OUT_DIR)

print("✔ Using predictions CSV:", PRED_CSV)
print("✔ Using recovery CSV:", REC_CSV)
print("✔ Using nodes CSV:", NODES_CSV)
print("✔ Using map image:", MAP_IMG)


# =====================================================
# FIGURE 1 — Track evolution
# =====================================================
def make_fig1_tracks():
    df = pd.read_csv(PRED_CSV)

    model_col = find_first_col(df, ["model", "Model", "MODEL"], name="pred")
    lead_col  = find_first_col(df, ["lead_hours", "lead_time", "horizon_hours", "leadtime_hours"], name="pred")

    gt_lat = find_first_col(df, ["gt_lat", "true_lat", "actual_lat", "y_true_lat"], name="pred")
    gt_lon = find_first_col(df, ["gt_lon", "true_lon", "actual_lon", "y_true_lon"], name="pred")
    pr_lat = find_first_col(df, ["pred_lat", "y_pred_lat", "forecast_lat", "lat_pred"], name="pred")
    pr_lon = find_first_col(df, ["pred_lon", "y_pred_lon", "forecast_lon", "lon_pred"], name="pred")

    case_col = find_first_col(df, ["storm_id", "case_id", "sequence_id", "track_id", "sample_id", "id"], required=False, name="pred")

    df[lead_col] = ensure_numeric(df[lead_col])
    for c in [gt_lat, gt_lon, pr_lat, pr_lon]:
        df[c] = ensure_numeric(df[c])

    if case_col is not None:
        chosen_case = df.groupby(case_col).size().sort_values(ascending=False).index[0]
        dcase = df[df[case_col] == chosen_case].copy()
    else:
        chosen_case = None
        dcase = df.copy()

    dcase = dcase.dropna(subset=[lead_col]).sort_values(lead_col)

    actual_by_lead = (
        dcase[[lead_col, gt_lat, gt_lon]]
        .dropna()
        .drop_duplicates(subset=[lead_col])
        .sort_values(lead_col)
    )
    lead = actual_by_lead[lead_col].to_numpy()
    gt_track = np.column_stack([actual_by_lead[gt_lat].to_numpy(), actual_by_lead[gt_lon].to_numpy()])

    wanted = ["LSTM", "Transformer", "GNO+DynGNN", "DynGNN", "GNO"]
    present = list(dcase[model_col].dropna().unique())
    ordered = [m for m in wanted if m in present]
    for m in present:
        if m not in ordered:
            ordered.append(m)
    ordered = ordered[:3]

    pred_tracks = {}
    for m in ordered:
        dm = dcase[dcase[model_col] == m].dropna(subset=[lead_col, pr_lat, pr_lon]).sort_values(lead_col)
        dm = dm.drop_duplicates(subset=[lead_col])
        merged = pd.merge(pd.DataFrame({lead_col: lead}), dm[[lead_col, pr_lat, pr_lon]], on=lead_col, how="left")
        pred_tracks[m] = np.column_stack([merged[pr_lat].to_numpy(), merged[pr_lon].to_numpy()])

    bg = mpimg.imread(MAP_IMG)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(bg, extent=MAP_EXTENT, origin="upper", aspect="auto", zorder=0)
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.grid(True, linestyle=":", color="deepskyblue", alpha=0.7)

    ax.plot([], [], color="cyan", lw=3, marker="o", label="Actual")
    gt_lat_arr = gt_track[:, 0]
    gt_lon_arr = gt_track[:, 1]
    if SHOW_ARROWS:
        add_arrows(ax, gt_lon_arr, gt_lat_arr, "cyan", lw=2.2)
    if SHOW_TIME_LABELS:
        add_time_labels(ax, gt_lon_arr, gt_lat_arr, lead, "cyan")

    colors = ["orange", "lime", "red"]
    for i, m in enumerate(ordered):
        tr = pred_tracks[m]
        ax.plot([], [], color=colors[i], lw=3, marker="o", label=m)
        if SHOW_ARROWS:
            add_arrows(ax, tr[:, 1], tr[:, 0], colors[i], lw=2.0)
        if SHOW_TIME_LABELS:
            add_time_labels(ax, tr[:, 1], tr[:, 0], lead, colors[i])

    title = "Figure 1 — Forecast Track Evolution Across Lead Times"
    if chosen_case is not None:
        title += f" (case: {chosen_case})"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(FIG1_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved:", FIG1_OUT)


# =====================================================
# FIGURE 2 — Recovery map
# =====================================================
def make_fig2_recovery():
    rec = pd.read_csv(REC_CSV)
    nodes = pd.read_csv(NODES_CSV)

    rec_node = find_first_col(rec, ["node_id", "FIPS", "fips", "node", "id"], required=True, name="recovery")
    rec_time = find_first_col(rec, ["t", "time", "time_step", "step", "timestamp"], required=False, name="recovery")
    rec_pred = find_first_col(rec, ["pred_recovery", "recovery_pred", "y_pred", "prediction", "pred"], required=True, name="recovery")

    node_id = find_first_col(nodes, ["FIPS", "fips", "node_id", "id"], required=True, name="nodes")
    lat_col = find_first_col(nodes, ["lat", "latitude", "LAT"], required=True, name="nodes")
    lon_col = find_first_col(nodes, ["lon", "longitude", "LON", "lng"], required=True, name="nodes")
    vuln_col = find_first_col(nodes, ["RPL_THEMES", "rpl_themes", "SVI", "svi"], required=False, name="nodes")

    rec[rec_pred] = ensure_numeric(rec[rec_pred])
    if rec_time is not None:
        rec[rec_time] = ensure_numeric(rec[rec_time])

    nodes[lat_col] = ensure_numeric(nodes[lat_col])
    nodes[lon_col] = ensure_numeric(nodes[lon_col])
    if vuln_col is not None:
        nodes[vuln_col] = ensure_numeric(nodes[vuln_col])

    rec_small = rec[[rec_node, rec_pred] + ([rec_time] if rec_time is not None else [])].copy()
    rec_small = rec_small.rename(columns={rec_node: "NodeID", rec_pred: "PredRecovery"})

    if rec_time is not None and pd.api.types.is_numeric_dtype(rec_small[rec_time]):
        idx = rec_small.groupby("NodeID")[rec_time].idxmax()
        rec_latest = rec_small.loc[idx].copy()
    else:
        rec_latest = rec_small.groupby("NodeID", as_index=False).tail(1)

    nodes_small = nodes.rename(columns={node_id: "NodeID"})
    merged = rec_latest.merge(nodes_small, on="NodeID", how="inner")

    bg = mpimg.imread(MAP_IMG)

    # zoom
    lon_min = float(np.nanmin(merged[lon_col])) - 5
    lon_max = float(np.nanmax(merged[lon_col])) + 5
    lat_min = float(np.nanmin(merged[lat_col])) - 5
    lat_max = float(np.nanmax(merged[lat_col])) + 5

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(bg, extent=MAP_EXTENT, origin="upper", aspect="auto", zorder=0)
    ax.set_xlim(max(MAP_EXTENT[0], lon_min), min(MAP_EXTENT[1], lon_max))
    ax.set_ylim(max(MAP_EXTENT[2], lat_min), min(MAP_EXTENT[3], lat_max))
    ax.grid(True, linestyle=":", color="deepskyblue", alpha=0.6)

    sc = ax.scatter(
        merged[lon_col], merged[lat_col],
        c=merged["PredRecovery"],
        s=18,
        alpha=0.9,
        zorder=5
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Predicted Recovery Index")

    if vuln_col is not None:
        hv = merged[merged[vuln_col] > 0.75]
        if len(hv) > 0:
            ax.scatter(hv[lon_col], hv[lat_col], s=35, facecolors="none",
                       edgecolors="yellow", linewidths=1.2, zorder=6)
            ax.plot([], [], color="yellow", marker="o", linestyle="None",
                    label="High vulnerability (outline)")

    ax.set_title("Figure 2 — Recovery Dynamics Under Context (Predicted Recovery)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(FIG2_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved:", FIG2_OUT)


def main():
    make_fig1_tracks()
    make_fig2_recovery()


if __name__ == "__main__":
    main()
