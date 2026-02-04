# =============================================================================
# Save as (VS Code-ready):
#   report_results.py
#
# What this script does
# 1) Reads your inference outputs:
#    - metrics/inference_test_metrics_summary.csv
#    - metrics/inference_test_predictions_all_models.csv
#
# 2) Produces:
#    - metrics/REPORT.md              (clean summary you can send to professor)
#    - metrics/plots/*.png            (publication-style plots)
#
# Requirements:
#   pip install pandas numpy matplotlib
#
# Run:
#   python report_results.py --metrics_dir /content/metrics
#   OR (Windows example):
#   python report_results.py --metrics_dir "C:\Users\Adrija\Downloads\DFGCN\metrics"
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LEADS = [6, 12, 24, 48]  # keep consistent with cfg.lead_hours


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_inputs(metrics_dir: str):
    summary_path = os.path.join(metrics_dir, "inference_test_metrics_summary.csv")
    preds_path = os.path.join(metrics_dir, "inference_test_predictions_all_models.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary metrics file: {summary_path}")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    summary = pd.read_csv(summary_path)
    preds = pd.read_csv(preds_path)
    return summary, preds


def ensure_plot_dir(metrics_dir: str) -> str:
    plot_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def compute_track_errors_from_preds(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-lead haversine errors for each row (model x sample).
    Returns long-form dataframe: [model, storm_tag, lead_h, err_km]
    """
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    rows = []
    for lead_h in LEADS:
        tlat = f"true_lat_{lead_h}h"
        tlon = f"true_lon_{lead_h}h"
        plat = f"pred_mu_lat_{lead_h}h"
        plon = f"pred_mu_lon_{lead_h}h"

        if not all(c in preds.columns for c in [tlat, tlon, plat, plon]):
            continue

        lat_true = preds[tlat].astype(float).values
        lon_true = preds[tlon].astype(float).values
        lat_pred = preds[plat].astype(float).values
        lon_pred = preds[plon].astype(float).values

        err = haversine_km(lat_true, lon_true, lat_pred, lon_pred)

        tmp = pd.DataFrame({
            "model": preds["model"].astype(str),
            "storm_tag": preds.get("storm_tag", "unknown").astype(str),
            "lead_h": lead_h,
            "err_km": err.astype(float),
        })
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["model", "storm_tag", "lead_h", "err_km"])
    return pd.concat(rows, ignore_index=True)


def compute_cone_coverage_from_preds(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes cone coverage from prediction CSV using the same axis-aligned ellipse criterion.
    P50/P90 approximate z for chi-square(2): sqrt(1.386)=1.177, sqrt(4.605)=2.146
    Returns long-form: [model, storm_tag, lead_h, cov50, cov90] with per-sample booleans.
    """
    Z_P50 = 1.177
    Z_P90 = 2.146

    out = []
    for lead_h in LEADS:
        tlat = f"true_lat_{lead_h}h"
        tlon = f"true_lon_{lead_h}h"
        mlat = f"pred_mu_lat_{lead_h}h"
        mlon = f"pred_mu_lon_{lead_h}h"
        slat = f"pred_sigma_lat_{lead_h}h"
        slon = f"pred_sigma_lon_{lead_h}h"

        if not all(c in preds.columns for c in [tlat, tlon, mlat, mlon, slat, slon]):
            continue

        lat_true = preds[tlat].astype(float).values
        lon_true = preds[tlon].astype(float).values
        mu_lat = preds[mlat].astype(float).values
        mu_lon = preds[mlon].astype(float).values
        sg_lat = preds[slat].astype(float).values
        sg_lon = preds[slon].astype(float).values

        # ellipse inclusion: ((x-mu)/sigma)^2 sum <= z^2
        dx = (lat_true - mu_lat) / (sg_lat + 1e-6)
        dy = (lon_true - mu_lon) / (sg_lon + 1e-6)
        q = dx * dx + dy * dy

        cov50 = (q <= (Z_P50 ** 2))
        cov90 = (q <= (Z_P90 ** 2))

        tmp = pd.DataFrame({
            "model": preds["model"].astype(str),
            "storm_tag": preds.get("storm_tag", "unknown").astype(str),
            "lead_h": lead_h,
            "cov50": cov50.astype(int),
            "cov90": cov90.astype(int),
        })
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=["model", "storm_tag", "lead_h", "cov50", "cov90"])
    return pd.concat(out, ignore_index=True)


def summarize_from_summary_csv(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a clean table with key metrics:
      - mean track error over leads
      - landfall_time_err_hours
      - cone coverages
    """
    # compute mean over lead track errors
    rows = []
    for _, r in summary.iterrows():
        model = str(r.get("model", "unknown"))
        track_vals = []
        for h in LEADS:
            track_vals.append(_safe_float(r.get(f"track_km_{h}h")))
        mean_track = float(np.nanmean(track_vals)) if np.any(np.isfinite(track_vals)) else np.nan

        rows.append({
            "model": model,
            "mean_track_km": mean_track,
            "track_6h_km": _safe_float(r.get("track_km_6h")),
            "track_12h_km": _safe_float(r.get("track_km_12h")),
            "track_24h_km": _safe_float(r.get("track_km_24h")),
            "track_48h_km": _safe_float(r.get("track_km_48h")),
            "landfall_time_err_hours": _safe_float(r.get("landfall_time_err_hours")),
            "cone_cov50_24h": _safe_float(r.get("cone_cov50_24h")),
            "cone_cov90_24h": _safe_float(r.get("cone_cov90_24h")),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("mean_track_km", ascending=True, na_position="last").reset_index(drop=True)
    return out


def plot_track_error_curve(summary: pd.DataFrame, plot_dir: str):
    """
    Plot: track error (km) vs lead time for each model (from summary csv).
    """
    plt.figure()
    for _, r in summary.iterrows():
        model = str(r.get("model", "unknown"))
        ys = []
        xs = []
        for h in LEADS:
            v = _safe_float(r.get(f"track_km_{h}h"))
            if np.isfinite(v):
                xs.append(h)
                ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=model)

    plt.xlabel("Lead time (hours)")
    plt.ylabel("Track error (km)")
    plt.title("Track error vs lead time (test split)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(plot_dir, "track_error_vs_lead.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_cone_coverage(summary: pd.DataFrame, plot_dir: str):
    """
    Plot: cone coverage vs lead time (P50 and P90) per model.
    """
    # P50
    plt.figure()
    for _, r in summary.iterrows():
        model = str(r.get("model", "unknown"))
        xs, ys = [], []
        for h in LEADS:
            v = _safe_float(r.get(f"cone_cov50_{h}h"))
            if np.isfinite(v):
                xs.append(h); ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Cone coverage (P50)")
    plt.title("Uncertainty calibration: P50 coverage (test split)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out50 = os.path.join(plot_dir, "cone_coverage_p50.png")
    plt.savefig(out50, dpi=200, bbox_inches="tight")
    plt.close()

    # P90
    plt.figure()
    for _, r in summary.iterrows():
        model = str(r.get("model", "unknown"))
        xs, ys = [], []
        for h in LEADS:
            v = _safe_float(r.get(f"cone_cov90_{h}h"))
            if np.isfinite(v):
                xs.append(h); ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Cone coverage (P90)")
    plt.title("Uncertainty calibration: P90 coverage (test split)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out90 = os.path.join(plot_dir, "cone_coverage_p90.png")
    plt.savefig(out90, dpi=200, bbox_inches="tight")
    plt.close()

    return out50, out90


def plot_landfall_error_bar(summary: pd.DataFrame, plot_dir: str):
    """
    Bar plot: landfall time error (hours) per model (from summary csv).
    """
    df = summary.copy()
    if "landfall_time_err_hours" not in df.columns:
        return None

    df = df[["model", "landfall_time_err_hours"]].copy()
    df["landfall_time_err_hours"] = pd.to_numeric(df["landfall_time_err_hours"], errors="coerce")

    plt.figure()
    plt.bar(df["model"].astype(str), df["landfall_time_err_hours"].values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Landfall time error (hours)")
    plt.title("Landfall time error proxy (Florida bbox)")
    plt.grid(True, axis="y", alpha=0.3)

    out = os.path.join(plot_dir, "landfall_time_error.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_track_error_boxplot(pred_err_long: pd.DataFrame, plot_dir: str):
    """
    Boxplot: distribution of per-sample track error at 24h and 48h by model.
    """
    keep = pred_err_long[pred_err_long["lead_h"].isin([24, 48])].copy()
    if keep.empty:
        return None

    for lead_h in [24, 48]:
        df = keep[keep["lead_h"] == lead_h].copy()
        if df.empty:
            continue

        plt.figure()
        models = sorted(df["model"].unique().tolist())
        data = [df[df["model"] == m]["err_km"].values for m in models]
        plt.boxplot(data, labels=models, showfliers=False)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Track error (km)")
        plt.title(f"Per-sample track error distribution at {lead_h}h (test split)")
        plt.grid(True, axis="y", alpha=0.3)
        out = os.path.join(plot_dir, f"boxplot_track_error_{lead_h}h.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()

    return True


def build_report_md(
    metrics_dir: str,
    summary_csv: pd.DataFrame,
    key_table: pd.DataFrame,
    plot_files: dict,
    pred_err_long: pd.DataFrame,
    cov_long: pd.DataFrame,
):
    """
    Writes a clean REPORT.md that references generated plots.
    """
    report_path = os.path.join(metrics_dir, "REPORT.md")

    # Determine best model by mean_track_km (lowest)
    best_row = key_table.iloc[0] if len(key_table) else None
    best_model = str(best_row["model"]) if best_row is not None else "N/A"
    best_mean = _safe_float(best_row["mean_track_km"]) if best_row is not None else np.nan

    # Optional per-storm breakdown (from predictions)
    storm_breakdown_txt = ""
    if not pred_err_long.empty and "storm_tag" in pred_err_long.columns:
        storm_breakdown = (
            pred_err_long.groupby(["model", "storm_tag", "lead_h"])["err_km"]
            .mean()
            .reset_index()
        )
        # keep only 24/48h for compactness
        storm_breakdown = storm_breakdown[storm_breakdown["lead_h"].isin([24, 48])]
        # pivot
        storm_pivot = storm_breakdown.pivot_table(
            index=["model", "storm_tag"], columns="lead_h", values="err_km", aggfunc="mean"
        ).reset_index()
        storm_pivot.columns = ["model", "storm_tag"] + [f"err_km_{int(c)}h" for c in storm_pivot.columns[2:]]
        storm_breakdown_txt = storm_pivot.to_markdown(index=False)

    # Optional coverage recompute check
    cov_txt = ""
    if not cov_long.empty:
        cov_mean = (
            cov_long.groupby(["model", "lead_h"])[["cov50", "cov90"]]
            .mean()
            .reset_index()
        )
        cov_piv = cov_mean.pivot_table(index="model", columns="lead_h", values=["cov50", "cov90"])
        # flatten columns
        cov_piv.columns = [f"{a}_@{b}h" for a, b in cov_piv.columns]
        cov_piv = cov_piv.reset_index()
        cov_txt = cov_piv.to_markdown(index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Experimental Results Report (Track Forecasting)\n\n")
        f.write("## Setup\n")
        f.write("- Task: Hurricane track forecasting (Irma 2017 + Ian 2022)\n")
        f.write("- Split: 80% train / 20% test (seeded)\n")
        f.write("- Outputs: probabilistic mean track + uncertainty (P50/P90 cone)\n\n")

        f.write("## Models evaluated\n")
        f.write("- Persistence (constant-velocity)\n")
        f.write("- LSTM baseline (past track + ERA5 patch)\n")
        f.write("- Transformer baseline (past track + ERA5 patch)\n")
        f.write("- Primary: GNO+DynGNN (operator-style ERA5 encoder + dynamic GNN)\n\n")

        f.write("## Key summary (from inference_test_metrics_summary.csv)\n")
        f.write(f"**Best mean track error:** {best_model} (mean={best_mean:.2f} km across 6/12/24/48h)\n\n")
        f.write(key_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Visualizations\n")
        if "track_curve" in plot_files:
            f.write(f"- Track error vs lead: `plots/{os.path.basename(plot_files['track_curve'])}`\n")
        if "cone50" in plot_files:
            f.write(f"- P50 cone coverage: `plots/{os.path.basename(plot_files['cone50'])}`\n")
        if "cone90" in plot_files:
            f.write(f"- P90 cone coverage: `plots/{os.path.basename(plot_files['cone90'])}`\n")
        if "landfall" in plot_files and plot_files["landfall"] is not None:
            f.write(f"- Landfall time error proxy: `plots/{os.path.basename(plot_files['landfall'])}`\n")
        f.write("\n")

        f.write("## Notes / Interpretation (fill in after you inspect plots)\n")
        f.write("- Persistence should degrade quickly beyond 12h.\n")
        f.write("- LSTM and Transformer should improve accuracy over Persistence.\n")
        f.write("- GNO+DynGNN should generally perform best at longer horizons and give smoother uncertainty.\n\n")

        if storm_breakdown_txt:
            f.write("## Per-storm breakdown (24h/48h mean error from predictions CSV)\n\n")
            f.write(storm_breakdown_txt)
            f.write("\n\n")

        if cov_txt:
            f.write("## Cone coverage recomputed from predictions CSV (mean)\n\n")
            f.write(cov_txt)
            f.write("\n\n")

        f.write("## Files used\n")
        f.write("- `inference_test_metrics_summary.csv`\n")
        f.write("- `inference_test_predictions_all_models.csv`\n")

    return report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Path to your metrics directory (contains inference_test_metrics_summary.csv and inference_test_predictions_all_models.csv)"
    )
    args = ap.parse_args()

    metrics_dir = args.metrics_dir
    summary, preds = load_inputs(metrics_dir)
    plot_dir = ensure_plot_dir(metrics_dir)

    # Summaries
    key_table = summarize_from_summary_csv(summary)

    # Derived analytics from predictions (more detailed)
    pred_err_long = compute_track_errors_from_preds(preds)
    cov_long = compute_cone_coverage_from_preds(preds)

    # Plots
    plot_files = {}
    plot_files["track_curve"] = plot_track_error_curve(summary, plot_dir)
    cone50, cone90 = plot_cone_coverage(summary, plot_dir)
    plot_files["cone50"] = cone50
    plot_files["cone90"] = cone90
    plot_files["landfall"] = plot_landfall_error_bar(summary, plot_dir)
    plot_track_error_boxplot(pred_err_long, plot_dir)

    # Report
    report_path = build_report_md(
        metrics_dir=metrics_dir,
        summary_csv=summary,
        key_table=key_table,
        plot_files=plot_files,
        pred_err_long=pred_err_long,
        cov_long=cov_long
    )

    print("\nDONE ✅")
    print("Report:", report_path)
    print("Plots:", plot_dir)
    print("Tip: Open REPORT.md and attach plots/ images when emailing your professor.\n")


if __name__ == "__main__":
    main()
