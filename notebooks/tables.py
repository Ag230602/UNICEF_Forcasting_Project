"""
Generate proposal tables from:
  1) inference_test_predictions_all_models.csv
  2) pred_recovery.csv
  3) nodes.csv

Outputs:
  outputs/tables/Table1_track_error_by_model_lead.csv
  outputs/tables/Table2_fde_by_model.csv
  outputs/tables/Table3_uncertainty_summary.csv
  outputs/tables/Table4_recovery_time_summary.csv
  outputs/tables/Table5_node_context_summary.csv
  outputs/tables/Table6_high_risk_nodes.csv

Run:
  python make_tables.py
"""

import os
import numpy as np
import pandas as pd


# =========================
# PATHS (edit if needed)
# =========================
PRED_PATH = r"C:\Users\Adrija\Downloads\DFGCN\metrics\inference_test_predictions_all_models.csv"
REC_PATH  = r"C:\Users\Adrija\Downloads\DFGCN\recovery_dataset_out\pred_recovery.csv"
NODES_PATH = r"C:\Users\Adrija\Downloads\DFGCN\recovery_dataset_out\nodes.csv"

OUT_DIR = os.path.join("outputs", "tables")


# =========================
# HELPERS
# =========================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def require_cols(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{df_name}] Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def find_first_col(df: pd.DataFrame, candidates: list[str], df_name: str, required=True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"[{df_name}] Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in km.
    lat/lon in degrees.
    """
    R = 6371.0088
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# =========================
# MAIN
# =========================
def main():
    ensure_out_dir(OUT_DIR)

    # -------------------------
    # Load CSVs
    # -------------------------
    pred = pd.read_csv(PRED_PATH)
    rec = pd.read_csv(REC_PATH)
    nodes = pd.read_csv(NODES_PATH)

    # -------------------------
    # Resolve column names (robust to minor variations)
    # -------------------------
    model_col = find_first_col(pred, ["model", "Model", "MODEL"], "predictions")

    lead_col  = find_first_col(pred, ["lead_hours", "lead_time", "horizon_hours", "leadtime_hours", "lead_hr"], "predictions")

    gt_lat_col = find_first_col(pred, ["gt_lat", "true_lat", "actual_lat", "y_true_lat", "lat_gt"], "predictions")
    gt_lon_col = find_first_col(pred, ["gt_lon", "true_lon", "actual_lon", "y_true_lon", "lon_gt"], "predictions")

    pr_lat_col = find_first_col(pred, ["pred_lat", "y_pred_lat", "forecast_lat", "lat_pred"], "predictions")
    pr_lon_col = find_first_col(pred, ["pred_lon", "y_pred_lon", "forecast_lon", "lon_pred"], "predictions")

    # Optional uncertainty cols (only if present)
    sig_lat_col = find_first_col(pred, ["sigma_lat", "std_lat", "pred_std_lat", "unc_lat"], "predictions", required=False)
    sig_lon_col = find_first_col(pred, ["sigma_lon", "std_lon", "pred_std_lon", "unc_lon"], "predictions", required=False)

    # Optional storm/sequence identifier (for FDE; else fallback)
    storm_id_col = find_first_col(
        pred,
        ["storm_id", "case_id", "sequence_id", "track_id", "sample_id", "id"],
        "predictions",
        required=False
    )

    # For recovery table
    rec_time_col = find_first_col(rec, ["t", "time", "time_step", "step", "timestamp"], "pred_recovery", required=False)
    rec_node_col = find_first_col(rec, ["node_id", "FIPS", "fips", "node", "id"], "pred_recovery", required=False)
    rec_pred_col = find_first_col(rec, ["pred_recovery", "recovery_pred", "y_pred", "prediction", "pred"], "pred_recovery")

    # For nodes context table
    nodes_id_col = find_first_col(nodes, ["FIPS", "fips", "node_id", "id"], "nodes", required=False)
    vuln_col = find_first_col(nodes, ["RPL_THEMES", "rpl_themes", "SVI", "svi"], "nodes", required=False)

    # Facility count candidates (use what exists)
    shelter_col = find_first_col(nodes, ["shelter_count", "Shelter_Count", "num_shelters", "shelters"], "nodes", required=False)
    hosp_col    = find_first_col(nodes, ["hospital_count", "Hospital_Count", "num_hospitals", "hospitals"], "nodes", required=False)
    school_col  = find_first_col(nodes, ["school_count", "School_Count", "num_schools", "schools"], "nodes", required=False)
    node_type_col = find_first_col(nodes, ["node_type", "type", "NodeType", "category"], "nodes", required=False)

    # -------------------------
    # Compute per-row track error (km)
    # -------------------------
    pred = pred.copy()
    pred["track_error_km"] = haversine_km(
        pred[gt_lat_col], pred[gt_lon_col],
        pred[pr_lat_col], pred[pr_lon_col]
    )

    # =========================
    # Table 1: Track error by model & lead time
    # =========================
    table1 = (
        pred
        .groupby([model_col, lead_col], dropna=False)["track_error_km"]
        .agg(mean_km="mean", median_km="median", n="count")
        .reset_index()
        .rename(columns={
            model_col: "Model",
            lead_col: "LeadTime_hours",
            "mean_km": "MeanTrackError_km",
            "median_km": "MedianTrackError_km",
            "n": "N"
        })
        .sort_values(["Model", "LeadTime_hours"])
    )
    table1_path = os.path.join(OUT_DIR, "Table1_track_error_by_model_lead.csv")
    table1.to_csv(table1_path, index=False)

    # =========================
    # Table 2: FDE by model
    # =========================
    if storm_id_col is not None:
        # For each (storm_id, model), keep the row at the maximum lead time
        idx = (
            pred
            .groupby([storm_id_col, model_col])[lead_col]
            .idxmax()
            .dropna()
            .astype(int)
        )
        fde_rows = pred.loc[idx].copy()
        fde_rows["is_fde_row"] = True
    else:
        # Fallback: use maximum lead time per model across entire file
        max_lead_by_model = pred.groupby(model_col)[lead_col].transform("max")
        fde_rows = pred[pred[lead_col] == max_lead_by_model].copy()

    table2 = (
        fde_rows
        .groupby(model_col, dropna=False)
        .agg(
            MaxLeadTime_hours=(lead_col, "max"),
            MeanFDE_km=("track_error_km", "mean"),
            MedianFDE_km=("track_error_km", "median"),
            N=("track_error_km", "count")
        )
        .reset_index()
        .rename(columns={model_col: "Model"})
        .sort_values("Model")
    )
    table2_path = os.path.join(OUT_DIR, "Table2_fde_by_model.csv")
    table2.to_csv(table2_path, index=False)

    # =========================
    # Table 3: Uncertainty summary (if sigma columns exist)
    # =========================
    if sig_lat_col is not None and sig_lon_col is not None:
        pred["uncertainty_proxy"] = pred[sig_lat_col].astype(float) * pred[sig_lon_col].astype(float)

        table3 = (
            pred
            .groupby(model_col, dropna=False)
            .agg(
                MeanSigmaLat=(sig_lat_col, "mean"),
                MeanSigmaLon=(sig_lon_col, "mean"),
                MeanUncertaintyProxy=("uncertainty_proxy", "mean"),
                MedianUncertaintyProxy=("uncertainty_proxy", "median"),
                N=("uncertainty_proxy", "count"),
            )
            .reset_index()
            .rename(columns={model_col: "Model"})
            .sort_values("Model")
        )
    else:
        # If no sigma columns, output a small note-style table derived from availability
        table3 = pd.DataFrame([{
            "Model": "N/A",
            "MeanSigmaLat": np.nan,
            "MeanSigmaLon": np.nan,
            "MeanUncertaintyProxy": np.nan,
            "MedianUncertaintyProxy": np.nan,
            "N": 0,
            "Note": "sigma_lat / sigma_lon not found in predictions CSV"
        }])

    table3_path = os.path.join(OUT_DIR, "Table3_uncertainty_summary.csv")
    table3.to_csv(table3_path, index=False)

    # =========================
    # Table 4: Recovery summary by time (if time exists) else overall
    # =========================
    rec = rec.copy()
    rec[rec_pred_col] = pd.to_numeric(rec[rec_pred_col], errors="coerce")

    if rec_time_col is not None:
        table4 = (
            rec
            .groupby(rec_time_col, dropna=False)[rec_pred_col]
            .agg(mean="mean", min="min", max="max", median="median", n="count")
            .reset_index()
            .rename(columns={
                rec_time_col: "TimeStep",
                "mean": "MeanPredRecovery",
                "min": "MinPredRecovery",
                "max": "MaxPredRecovery",
                "median": "MedianPredRecovery",
                "n": "N"
            })
            .sort_values("TimeStep")
        )
    else:
        table4 = pd.DataFrame([{
            "TimeStep": "N/A",
            "MeanPredRecovery": rec[rec_pred_col].mean(),
            "MinPredRecovery": rec[rec_pred_col].min(),
            "MaxPredRecovery": rec[rec_pred_col].max(),
            "MedianPredRecovery": rec[rec_pred_col].median(),
            "N": rec[rec_pred_col].notna().sum(),
            "Note": "No time column found; aggregated across all rows"
        }])

    table4_path = os.path.join(OUT_DIR, "Table4_recovery_time_summary.csv")
    table4.to_csv(table4_path, index=False)

    # =========================
    # Table 5: Node context summary (by node type if available; else overall)
    # =========================
    nodes = nodes.copy()

    # Build a list of columns that exist for context aggregation
    agg_cols = []
    if vuln_col is not None: agg_cols.append(vuln_col)
    if shelter_col is not None: agg_cols.append(shelter_col)
    if hosp_col is not None: agg_cols.append(hosp_col)
    if school_col is not None: agg_cols.append(school_col)

    if len(agg_cols) == 0:
        table5 = pd.DataFrame([{
            "Note": "No vulnerability/facility count columns found in nodes.csv",
            "AvailableColumns": ", ".join(nodes.columns)
        }])
    else:
        # numeric conversion where possible
        for c in agg_cols:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

        if node_type_col is not None:
            grp = nodes.groupby(node_type_col, dropna=False)
            table5 = grp[agg_cols].mean().reset_index().rename(columns={node_type_col: "NodeType"})
        else:
            # single-row summary
            means = {f"Mean_{c}": nodes[c].mean() for c in agg_cols}
            table5 = pd.DataFrame([means])

        # Rename to friendly headers
        rename_map = {}
        if vuln_col is not None: rename_map[vuln_col] = "Mean_RPL_THEMES_or_SVI"
        if shelter_col is not None: rename_map[shelter_col] = "Mean_ShelterCount"
        if hosp_col is not None: rename_map[hosp_col] = "Mean_HospitalCount"
        if school_col is not None: rename_map[school_col] = "Mean_SchoolCount"
        table5 = table5.rename(columns=rename_map)

    table5_path = os.path.join(OUT_DIR, "Table5_node_context_summary.csv")
    table5.to_csv(table5_path, index=False)

    # =========================
    # Table 6: High-risk nodes (requires merge of nodes + recovery + vulnerability)
    # =========================
    table6 = pd.DataFrame()

    if (rec_node_col is not None) and (nodes_id_col is not None) and (vuln_col is not None):
        # Prepare keys
        rec_key = rec_node_col
        nodes_key = nodes_id_col

        rec_small = rec[[rec_key, rec_pred_col]].copy()
        rec_small.columns = ["NodeID", "PredRecovery"]

        nodes_small_cols = [nodes_key, vuln_col]
        if shelter_col is not None: nodes_small_cols.append(shelter_col)
        if hosp_col is not None: nodes_small_cols.append(hosp_col)
        if school_col is not None: nodes_small_cols.append(school_col)

        nodes_small = nodes[nodes_small_cols].copy()
        nodes_small = nodes_small.rename(columns={nodes_key: "NodeID", vuln_col: "RPL_THEMES_or_SVI"})

        # numeric coercion
        nodes_small["RPL_THEMES_or_SVI"] = pd.to_numeric(nodes_small["RPL_THEMES_or_SVI"], errors="coerce")
        rec_small["PredRecovery"] = pd.to_numeric(rec_small["PredRecovery"], errors="coerce")

        merged = rec_small.merge(nodes_small, on="NodeID", how="inner")

        # If recovery has time, take latest time per node (most common)
        if rec_time_col is not None and rec_node_col in rec.columns:
            rec_latest = (
                rec[[rec_node_col, rec_time_col, rec_pred_col]]
                .rename(columns={rec_node_col: "NodeID", rec_pred_col: "PredRecovery", rec_time_col: "TimeStep"})
            )
            # pick max TimeStep per NodeID (if numeric)
            if pd.api.types.is_numeric_dtype(rec_latest["TimeStep"]):
                idx2 = rec_latest.groupby("NodeID")["TimeStep"].idxmax()
                rec_latest = rec_latest.loc[idx2]
            else:
                # if TimeStep is not numeric, just keep last row per NodeID as-is
                rec_latest = rec_latest.groupby("NodeID").tail(1)

            merged = rec_latest.merge(nodes_small, on="NodeID", how="inner")

        # Thresholds (proposal-safe; adjust if you want)
        high_vuln = merged["RPL_THEMES_or_SVI"] > 0.75
        low_recovery = merged["PredRecovery"] < 0.50

        table6 = merged[high_vuln & low_recovery].copy()

        # Sort by lowest recovery then highest vulnerability
        table6 = table6.sort_values(["PredRecovery", "RPL_THEMES_or_SVI"], ascending=[True, False])

        # Rename facility columns if present
        ren = {}
        if shelter_col is not None and shelter_col in table6.columns: ren[shelter_col] = "ShelterCount"
        if hosp_col is not None and hosp_col in table6.columns: ren[hosp_col] = "HospitalCount"
        if school_col is not None and school_col in table6.columns: ren[school_col] = "SchoolCount"
        table6 = table6.rename(columns=ren)

        # Keep a clean column order
        base_cols = ["NodeID"]
        if "TimeStep" in table6.columns: base_cols.append("TimeStep")
        base_cols += ["PredRecovery", "RPL_THEMES_or_SVI"]
        for c in ["ShelterCount", "HospitalCount", "SchoolCount"]:
            if c in table6.columns:
                base_cols.append(c)
        table6 = table6[base_cols]

    else:
        table6 = pd.DataFrame([{
            "Note": "Cannot build high-risk nodes table. Need: node id in pred_recovery + node id & vulnerability in nodes.csv",
            "pred_recovery_node_col_found": rec_node_col,
            "nodes_id_col_found": nodes_id_col,
            "vulnerability_col_found": vuln_col,
        }])

    table6_path = os.path.join(OUT_DIR, "Table6_high_risk_nodes.csv")
    table6.to_csv(table6_path, index=False)

    # -------------------------
    # Print paths
    # -------------------------
    print("\nSaved tables to:", os.path.abspath(OUT_DIR))
    print(" -", os.path.basename(table1_path))
    print(" -", os.path.basename(table2_path))
    print(" -", os.path.basename(table3_path))
    print(" -", os.path.basename(table4_path))
    print(" -", os.path.basename(table5_path))
    print(" -", os.path.basename(table6_path))


if __name__ == "__main__":
    main()
