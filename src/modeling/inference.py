import os
import numpy as np
import pandas as pd
import torch

from track_pipeline_unified_X import (
    CFG,
    parse_track,
    open_era5,
    build_samples,
    TrackDataset,
    GNO_DynGNN,
    LSTMTrackBaseline,
    TransformerTrackBaseline,
    evaluate_prob_model
)

cfg = CFG()
DEVICE = cfg.device


def build_train_test_split(samples, train_ratio=0.80, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_train = int(len(samples) * train_ratio)
    return idx[:n_train], idx[n_train:]


def rebuild_samples():
    irma_df = parse_track(cfg.irma_track, "irma")
    ian_df  = parse_track(cfg.ian_track,  "ian")

    irma_ds = open_era5(cfg.irma_era5)
    ian_ds  = open_era5(cfg.ian_era5)

    print("Rebuilding samples (Irma)...")
    s1 = build_samples(irma_df, irma_ds)
    print("Rebuilding samples (Ian)...")
    s2 = build_samples(ian_df, ian_ds)

    samples = s1 + s2
    print(f"Total samples rebuilt: {len(samples)}")
    return samples


def load_test_dataset_and_meta():
    samples = rebuild_samples()
    tr_idx, te_idx = build_train_test_split(samples, train_ratio=0.80, seed=cfg.seed)
    te_samples = [samples[i] for i in te_idx]
    print(f"Split: train={len(tr_idx)} test={len(te_idx)} (80/20)")
    return TrackDataset(te_samples), te_samples


def load_model(model_name: str):
    if model_name == "LSTM":
        model = LSTMTrackBaseline(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata
        )
        ckpt = os.path.join(cfg.ckpt_dir, "baseline_lstm.pt")

    elif model_name == "Transformer":
        model = TransformerTrackBaseline(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata
        )
        ckpt = os.path.join(cfg.ckpt_dir, "baseline_transformer.pt")

    elif model_name == "GNO+DynGNN":
        model = GNO_DynGNN(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata
        )
        ckpt = os.path.join(cfg.ckpt_dir, "main_gno_dyngnn.pt")

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["state"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, loader):
    """
    Returns a list of dict rows with:
      storm_tag, t0, lead_hours, gt_lat, gt_lon, pred_lat, pred_lon, sigma_lat, sigma_lon
    """
    rows = []
    for past, X, meta, y, info in loader:
        storm_tag, t0, lat0, lon0 = info  # tuples of length B
        past = past.to(DEVICE)
        X = X.to(DEVICE)
        meta = meta.to(DEVICE)

        mu, sigma = model(past, X, meta)  # (B,L,2)

        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
        y = y.numpy()

        B, L, _ = mu.shape
        for b in range(B):
            for li, h in enumerate(cfg.lead_hours):
                rows.append({
                    "storm_tag": storm_tag[b],
                    "t0": t0[b],
                    "lead_hours": int(h),
                    "gt_lat": float(y[b, li, 0]),
                    "gt_lon": float(y[b, li, 1]),
                    "pred_lat": float(mu[b, li, 0]),
                    "pred_lon": float(mu[b, li, 1]),
                    "sigma_lat": float(sigma[b, li, 0]),
                    "sigma_lon": float(sigma[b, li, 1]),
                })
    return rows


def main():
    os.makedirs(cfg.metrics_dir, exist_ok=True)

    print("Loading test dataset (rebuild + 80/20 split)...")
    te_ds, te_samples = load_test_dataset_and_meta()
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    # -------- metrics summary --------
    metric_rows = []
    # -------- predictions all models --------
    pred_rows_all = []

    for name in ["LSTM", "Transformer", "GNO+DynGNN"]:
        print(f"\nRunning inference: {name}")
        model = load_model(name)

        # metrics
        metrics = evaluate_prob_model(model, te_loader)
        metric_rows.append({"model": name, **metrics})
        print({k: round(v, 3) for k, v in metrics.items()})

        # predictions
        rows = predict_all(model, te_loader)
        for r in rows:
            r["model"] = name
        pred_rows_all.extend(rows)

    # write outputs expected by Summary.py
    out_metrics = os.path.join(cfg.metrics_dir, "inference_test_metrics_summary.csv")
    pd.DataFrame(metric_rows).to_csv(out_metrics, index=False)
    print("\nSaved:", out_metrics)

    out_preds = os.path.join(cfg.metrics_dir, "inference_test_predictions_all_models.csv")
    pd.DataFrame(pred_rows_all).to_csv(out_preds, index=False)
    print("Saved:", out_preds)


if __name__ == "__main__":
    main()

# python Summary.py --metrics_dir "C:\Users\Adrija\Downloads\DFGCN\metrics"
