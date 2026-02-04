# =============================================================================
# Save as (recommended):
#   C:\Users\Adrija\Downloads\DFGCN\track_pipeline_unified_X.py
#
# "Paper-ready" PATH-ONLY pipeline with:
#   ✅ Unified representation: X(t,node,modality,features)
#       node     = storm-centered grid points (GxG)
#       modality = ERA5 atmospheric
#       features = [u850,v850,u500,v500,z500]  (pressure fields can be added later)
#
#   ✅ Primary model:
#       Graph-Neural-Operator-style Encoder + Dynamic GNN + Probabilistic Head
#
#   ✅ Strong baselines for benchmarking (publication-connected):
#       - Persistence (constant velocity)
#       - LSTM baseline (past track sequence + ERA5 patch)
#       - Transformer baseline (past track sequence + ERA5 patch)
#       - HAFS hook (NOAA operational)   -> loader stub + evaluation interface
#       - Pangu/FourCastNet hook         -> loader stub + evaluation interface
#
#   ✅ Metrics (path-only, publication standard):
#       - Track error (km) at 6/12/24/48h
#       - Along-track / cross-track error (diagnostic approximation)
#       - Cone coverage P50/P90 (calibration; Gaussian ellipse approx)
#       - Landfall time error (proxy using Florida bounding box)
#
# Data expected:
#   Tracks (processed):
#     data\processed\tracks\irma_2017_hurdat2.csv
#     data\processed\tracks\ian_2022_hurdat2.csv
#   ERA5 NetCDF (raw):
#     data\raw\era5\irma_2017\era5_pl_irma_2017.nc
#     data\raw\era5\ian_2022\era5_pl_ian_2022.nc
#
# Install:
#   pip install numpy pandas xarray netCDF4 torch tqdm scikit-learn
#
# Run:
#   python "C:\Users\Adrija\Downloads\DFGCN\track_pipeline_unified_X.py"
# =============================================================================

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    base: str = r"C:\Users\Adrija\Downloads\DFGCN\data"

    irma_track: str = r"C:\Users\Adrija\Downloads\DFGCN\data\processed\tracks\irma_2017_hurdat2.csv"
    ian_track:  str = r"C:\Users\Adrija\Downloads\DFGCN\data\processed\tracks\ian_2022_hurdat2.csv"

    irma_era5: str = r"C:\Users\Adrija\Downloads\DFGCN\data\raw\era5\irma_2017\era5_pl_irma_2017.nc"
    ian_era5:  str = r"C:\Users\Adrija\Downloads\DFGCN\data\raw\era5\ian_2022\era5_pl_ian_2022.nc"

    # Unified representation X(t,node,modality,features)
    grid_size: int = 33          # node grid size: GxG
    crop_deg: float = 8.0        # storm-centered crop half-size in degrees
    features: Tuple[str, ...] = ("u850","v850","u500","v500","z500")

    history_steps: int = 4       # last N steps (each 6h)
    lead_hours: Tuple[int, ...] = (6, 12, 24, 48)

    include_metadata: bool = True  # vmax, mslp

    # Training (keep small for fast iteration; you can increase later)
    seed: int = 42
    batch_size: int = 16
    epochs_main: int = 20
    epochs_baseline: int = 12
    lr: float = 2e-4
    wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_root: str = r"C:\Users\Adrija\Downloads\DFGCN"
    ckpt_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\checkpoints"
    metrics_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\metrics"

    # Florida landfall proxy (simple bounding box)
    florida_bbox: Tuple[float, float, float, float] = (24.0, 32.0, -88.0, -79.0)  # (lat_min, lat_max, lon_min, lon_max)

    # If prediction/label never enters bbox within leads, landfall error is undefined
    landfall_missing_policy: str = "ignore"  # "ignore" or "max"

cfg = CFG()


# ----------------------------
# Repro helpers
# ----------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)


# ----------------------------
# Geodesy / metrics helpers
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def bearing_rad(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return math.atan2(y, x)


def along_cross_track_errors(lat0, lon0, lat_true, lon_true, lat_pred, lon_pred):
    """
    Approx diagnostic decomposition: along-track and cross-track.
    Uses bearing origin->true as the reference direction.
    """
    ref = bearing_rad(lat0, lon0, lat_true, lon_true)
    d_op = haversine_km(lat0, lon0, lat_pred, lon_pred)
    b_op = bearing_rad(lat0, lon0, lat_pred, lon_pred)
    dtheta = b_op - ref
    along = d_op * math.cos(dtheta)
    cross = d_op * math.sin(dtheta)
    d_ot = haversine_km(lat0, lon0, lat_true, lon_true)
    along_err = along - d_ot
    cross_err = cross
    return along_err, cross_err


def in_bbox(lat, lon, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def first_entry_lead_index(lat_seq, lon_seq, bbox):
    """
    Returns index in [0..L-1] of first entry into bbox, else None.
    lat_seq/lon_seq are sequences over lead steps only.
    """
    for i, (la, lo) in enumerate(zip(lat_seq, lon_seq)):
        if in_bbox(float(la), float(lo), bbox):
            return i
    return None


# ----------------------------
# ERA5 IO + robust crop (matches your data)
# ----------------------------
def open_era5(nc_path: str) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)

    # vars
    var_u = "u" if "u" in ds.variables else "u_component_of_wind"
    var_v = "v" if "v" in ds.variables else "v_component_of_wind"
    var_z = "z" if "z" in ds.variables else "geopotential"
    if var_u not in ds.variables or var_v not in ds.variables or var_z not in ds.variables:
        raise ValueError(f"ERA5 vars not found. ds.variables={list(ds.variables)}")

    # pressure coord
    if "level" in ds.coords:
        plev = "level"
    elif "pressure_level" in ds.coords:
        plev = "pressure_level"
    else:
        raise ValueError(f"Pressure level coord not found. coords={list(ds.coords)}")

    # time coord (your files can be valid_time)
    if "time" in ds.coords:
        tcoord = "time"
    elif "valid_time" in ds.coords:
        tcoord = "valid_time"
    else:
        raise ValueError(f"Time coord not found. coords={list(ds.coords)}")

    # Normalize longitude to [-180, 180] and sort (prevents empty crops)
    if "longitude" not in ds.coords:
        raise ValueError("ERA5 missing longitude coordinate")
    lon = ds["longitude"]
    if float(lon.max()) > 180:
        lon_new = ((lon + 180) % 360) - 180
        ds = ds.assign_coords(longitude=lon_new)
    ds = ds.sortby("longitude")

    ds.attrs["_u"] = var_u
    ds.attrs["_v"] = var_v
    ds.attrs["_z"] = var_z
    ds.attrs["_plev"] = plev
    ds.attrs["_tcoord"] = tcoord
    return ds


def parse_track(csv_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["storm_tag"] = tag
    for col in ["vmax_kt", "mslp_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def crop_era5_to_X(ds: xr.Dataset, tstamp: pd.Timestamp, lat0: float, lon0: float) -> np.ndarray:
    """
    Returns X(t,node,atmo,features) as (F,G,G) where
      F = [u850,v850,u500,v500,z500]
    """
    tcoord = ds.attrs["_tcoord"]
    uvar, vvar, zvar = ds.attrs["_u"], ds.attrs["_v"], ds.attrs["_z"]
    plev = ds.attrs["_plev"]

    dsel = ds.sel({tcoord: np.datetime64(tstamp.to_datetime64())}, method="nearest")

    lat_min, lat_max = lat0 - cfg.crop_deg, lat0 + cfg.crop_deg
    lon_min, lon_max = lon0 - cfg.crop_deg, lon0 + cfg.crop_deg

    # latitude slice direction-aware
    lat_vals = dsel["latitude"].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_slice = slice(lon_min, lon_max)  # longitude normalized to [-180,180] and sorted

    box = dsel.sel(latitude=lat_slice, longitude=lon_slice)

    if box.sizes.get("longitude", 0) == 0 or box.sizes.get("latitude", 0) == 0:
        raise RuntimeError(
            f"Empty crop at t={tstamp} lat0={lat0:.2f} lon0={lon0:.2f} "
            f"lat[{lat_min:.2f},{lat_max:.2f}] lon[{lon_min:.2f},{lon_max:.2f}] "
            f"got lat={box.sizes.get('latitude',0)} lon={box.sizes.get('longitude',0)}"
        )

    def pl(varname: str, level: int):
        return box[varname].sel({plev: level}).values.astype(np.float32)

    u850 = pl(uvar, 850); v850 = pl(vvar, 850)
    u500 = pl(uvar, 500); v500 = pl(vvar, 500)
    z500 = pl(zvar, 500)

    X = np.stack([u850, v850, u500, v500, z500], axis=0)  # (F,H,W)
    Xt = torch.from_numpy(X).unsqueeze(0)  # (1,F,H,W)
    Xt = F.interpolate(Xt, size=(cfg.grid_size, cfg.grid_size), mode="bilinear", align_corners=False)
    return Xt.squeeze(0).numpy()  # (F,G,G)


# ----------------------------
# Sample builder (Input/Output definition)
# ----------------------------
def build_samples(track_df: pd.DataFrame, era5_ds: xr.Dataset) -> List[Dict]:
    lead_steps = [h // 6 for h in cfg.lead_hours]
    samples = []
    skipped = 0

    for i in range(cfg.history_steps, len(track_df)):
        if i + max(lead_steps) >= len(track_df):
            break

        t0 = track_df.loc[i, "datetime_utc"]
        lat0 = float(track_df.loc[i, "lat"])
        lon0 = float(track_df.loc[i, "lon"])

        # past positions (H,2) oldest->newest
        past = []
        for k in range(cfg.history_steps, 0, -1):
            past.append([float(track_df.loc[i-k, "lat"]), float(track_df.loc[i-k, "lon"])])
        past = np.array(past, dtype=np.float32)

        # meta
        meta = None
        if cfg.include_metadata:
            vmax = float(track_df.loc[i, "vmax_kt"]) if pd.notna(track_df.loc[i, "vmax_kt"]) else 0.0
            mslp = float(track_df.loc[i, "mslp_mb"]) if pd.notna(track_df.loc[i, "mslp_mb"]) else 0.0
            meta = np.array([vmax, mslp], dtype=np.float32)

        # unified X(t,node,atmo,features)
        try:
            X = crop_era5_to_X(era5_ds, t0, lat0, lon0)
        except Exception:
            skipped += 1
            continue

        # labels: future absolute positions at leads
        y_abs = []
        for step in lead_steps:
            y_abs.append([float(track_df.loc[i+step, "lat"]), float(track_df.loc[i+step, "lon"])])
        y_abs = np.array(y_abs, dtype=np.float32)

        samples.append({
            "storm_tag": track_df.loc[i, "storm_tag"],
            "t0": t0.isoformat(),
            "past": past,
            "X": X,
            "meta": meta,
            "y_abs": y_abs,
            "lat0": lat0,
            "lon0": lon0
        })

    print(f"[build_samples:{track_df['storm_tag'].iloc[0]}] kept={len(samples)} skipped={skipped}")
    return samples


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        past = torch.from_numpy(s["past"])              # (H,2)
        X = torch.from_numpy(s["X"])                    # (F,G,G)
        y = torch.from_numpy(s["y_abs"])                # (L,2)
        meta = torch.from_numpy(s["meta"]) if s["meta"] is not None else torch.zeros(2)
        info = (s["storm_tag"], s["t0"], s["lat0"], s["lon0"])
        return past, X, meta, y, info


# ----------------------------
# Baseline HOOKS (HAFS / Pangu / FourCastNet)
# ----------------------------
def load_hafs_track(storm_tag: str) -> pd.DataFrame:
    """
    Hook for NOAA HAFS operational tracks.

    Expected return format (example columns):
      datetime_utc (timestamp), lead_hours (int), lat (float), lon (float)

    Implement later by downloading HAFS track products for Irma/Ian and parsing.
    For now, raise NotImplementedError so the pipeline remains "paper-ready"
    with a clear plug-in point.
    """
    raise NotImplementedError("HAFS loader not implemented yet. Add parser for NOAA HAFS track products.")


def load_pangu_or_fourcastnet_track(storm_tag: str) -> pd.DataFrame:
    """
    Hook for extracted tracks from Pangu-Weather/FourCastNet reanalysis-style outputs.

    Expected return format:
      datetime_utc (timestamp), lead_hours (int), lat (float), lon (float)

    Implement later: run/download the model output fields and extract a track
    (e.g., using minimum sea-level pressure center or vorticity center).
    """
    raise NotImplementedError("Pangu/FourCastNet track extraction not implemented yet.")


# ----------------------------
# Models
# ----------------------------
class PersistenceBaseline:
    """Constant velocity extrapolation using last two points in history."""
    def predict_np(self, past: np.ndarray, lead_steps: List[int]) -> np.ndarray:
        p1 = past[-2]
        p2 = past[-1]
        v = p2 - p1  # degrees per 6h
        preds = [p2 + v * s for s in lead_steps]
        return np.array(preds, dtype=np.float32)  # (L,2)


class OperatorEncoder(nn.Module):
    """
    Lightweight operator-style encoder (fast). For a true FNO later,
    swap this with an FNO/AFNO module without changing the pipeline.
    """
    def __init__(self, in_ch: int, width: int = 48, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, out_dim),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.head(self.net(X))


class DynamicGNN(nn.Module):
    """Dynamic message passing over history nodes (positions)."""
    def __init__(self, node_dim=32, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Linear(2, node_dim)
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, node_dim))
            for _ in range(layers)
        ])
        self.readout = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, past):
        # past: (B,H,2)
        h = self.embed(past)  # (B,H,node_dim)
        tau = 2.0
        d2 = ((past[:, :, None, :] - past[:, None, :, :]) ** 2).sum(-1)  # (B,H,H)
        A = torch.softmax(-d2 / tau, dim=-1)
        for layer in self.mlp:
            m = torch.einsum("bij,bjn->bin", A, h)
            h = h + layer(m)
        g = h.mean(dim=1)
        return self.readout(g)  # (B,hidden)


class ProbTrackHead(nn.Module):
    """Outputs Gaussian (mu, sigma) for each lead time (lat,lon)."""
    def __init__(self, in_dim: int, leads: int):
        super().__init__()
        self.leads = leads
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, leads * 2)
        self.log_sigma = nn.Linear(128, leads * 2)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h).view(-1, self.leads, 2)
        log_sigma = self.log_sigma(h).view(-1, self.leads, 2).clamp(-6, 3)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class GNO_DynGNN(nn.Module):
    """
    Primary model:
      Operator-style encoder over ERA5 patch + DynamicGNN over history + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=48, out_dim=128)
        self.gnn = DynamicGNN(node_dim=32, hidden=64, layers=2)
        self.past_mlp = nn.Sequential(nn.Flatten(), nn.Linear(cfg.history_steps * 2, 64), nn.ReLU())
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=128 + 64 + 64 + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)          # (B,128)
        g  = self.gnn(past)      # (B,64)
        p  = self.past_mlp(past) # (B,64)
        parts = [op, g, p]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


class LSTMTrackBaseline(nn.Module):
    """
    Baseline: LSTM over past positions + ERA5 operator encoder + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True, hidden: int = 64):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=32, out_dim=96)
        self.pos_embed = nn.Linear(2, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, num_layers=1, batch_first=True)
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=96 + hidden + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)  # (B,96)
        seq = self.pos_embed(past)  # (B,H,32)
        out, (hn, cn) = self.lstm(seq)
        h_last = hn[-1]  # (B,hidden)
        parts = [op, h_last]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


class TransformerTrackBaseline(nn.Module):
    """
    Baseline: Transformer encoder over past positions + ERA5 operator encoder + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=32, out_dim=96)
        self.pos_embed = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=96 + d_model + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)  # (B,96)
        seq = self.pos_embed(past)          # (B,H,d)
        z = self.tr(seq)                    # (B,H,d)
        pooled = z.mean(dim=1)              # (B,d)
        parts = [op, pooled]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


# ----------------------------
# Loss + cone coverage
# ----------------------------
def gaussian_nll(mu, sigma, y):
    eps = 1e-6
    var = sigma**2 + eps
    return 0.5 * torch.mean(((y - mu) ** 2) / var + torch.log(var))


def ellipse_inclusion(lat_true, lon_true, mu_lat, mu_lon, sigma_lat, sigma_lon, z):
    """
    Axis-aligned Gaussian ellipse inclusion:
      ((x-mu)/sigma)^2 sum <= z^2
    """
    dx = (lat_true - mu_lat) / (sigma_lat + 1e-6)
    dy = (lon_true - mu_lon) / (sigma_lon + 1e-6)
    return (dx*dx + dy*dy) <= (z*z)


# For 2D Gaussian: use chi-square cutoffs (approx)
Z_P50 = 1.177  # sqrt(chi2_2(0.50))
Z_P90 = 2.146  # sqrt(chi2_2(0.90))


# ----------------------------
# Evaluation (includes landfall time error)
# ----------------------------
@torch.no_grad()
def evaluate_prob_model(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()

    track_err = [[] for _ in cfg.lead_hours]
    along_err = [[] for _ in cfg.lead_hours]
    cross_err = [[] for _ in cfg.lead_hours]
    cov50 = [[] for _ in cfg.lead_hours]
    cov90 = [[] for _ in cfg.lead_hours]

    landfall_err_hours = []

    for past, X, meta, y, info in loader:
        past = past.to(cfg.device)
        X = X.to(cfg.device)
        meta = meta.to(cfg.device)
        y = y.to(cfg.device)

        mu, sigma = model(past, X, meta)  # (B,L,2)
        mu_lat, mu_lon = mu[..., 0], mu[..., 1]
        sig_lat, sig_lon = sigma[..., 0], sigma[..., 1]

        for b in range(mu.size(0)):
            lat0 = float(past[b, -1, 0].cpu())
            lon0 = float(past[b, -1, 1].cpu())

            # landfall proxy: among lead points only
            true_lat_seq = y[b, :, 0].cpu().numpy()
            true_lon_seq = y[b, :, 1].cpu().numpy()
            pred_lat_seq = mu_lat[b, :].cpu().numpy()
            pred_lon_seq = mu_lon[b, :].cpu().numpy()

            t_idx = first_entry_lead_index(true_lat_seq, true_lon_seq, cfg.florida_bbox)
            p_idx = first_entry_lead_index(pred_lat_seq, pred_lon_seq, cfg.florida_bbox)

            if t_idx is None and p_idx is None:
                pass
            elif t_idx is None or p_idx is None:
                if cfg.landfall_missing_policy == "max":
                    landfall_err_hours.append(float(max(cfg.lead_hours)))
            else:
                landfall_err_hours.append(abs(cfg.lead_hours[p_idx] - cfg.lead_hours[t_idx]))

            for li, h in enumerate(cfg.lead_hours):
                lat_t = float(y[b, li, 0].cpu())
                lon_t = float(y[b, li, 1].cpu())
                lat_p = float(mu_lat[b, li].cpu())
                lon_p = float(mu_lon[b, li].cpu())

                km = haversine_km(lat_t, lon_t, lat_p, lon_p)
                at, ct = along_cross_track_errors(lat0, lon0, lat_t, lon_t, lat_p, lon_p)

                track_err[li].append(km)
                along_err[li].append(at)
                cross_err[li].append(ct)

                cov50[li].append(bool(ellipse_inclusion(
                    lat_t, lon_t,
                    float(mu_lat[b, li].cpu()),
                    float(mu_lon[b, li].cpu()),
                    float(sig_lat[b, li].cpu()),
                    float(sig_lon[b, li].cpu()),
                    Z_P50
                )))
                cov90[li].append(bool(ellipse_inclusion(
                    lat_t, lon_t,
                    float(mu_lat[b, li].cpu()),
                    float(mu_lon[b, li].cpu()),
                    float(sig_lat[b, li].cpu()),
                    float(sig_lon[b, li].cpu()),
                    Z_P90
                )))

    metrics: Dict[str, float] = {}
    for i, h in enumerate(cfg.lead_hours):
        metrics[f"track_km_{h}h"] = float(np.mean(track_err[i])) if track_err[i] else float("nan")
        metrics[f"along_err_km_{h}h"] = float(np.mean(along_err[i])) if along_err[i] else float("nan")
        metrics[f"cross_err_km_{h}h"] = float(np.mean(cross_err[i])) if cross_err[i] else float("nan")
        metrics[f"cone_cov50_{h}h"] = float(np.mean(cov50[i])) if cov50[i] else float("nan")
        metrics[f"cone_cov90_{h}h"] = float(np.mean(cov90[i])) if cov90[i] else float("nan")

    metrics["landfall_time_err_hours"] = float(np.mean(landfall_err_hours)) if landfall_err_hours else float("nan")
    return metrics


def evaluate_persistence(te_ds: TrackDataset) -> Dict[str, float]:
    pers = PersistenceBaseline()
    lead_steps = [h // 6 for h in cfg.lead_hours]
    out: Dict[str, List[float]] = {f"track_km_{h}h": [] for h in cfg.lead_hours}

    landfall_err_hours = []

    for s in te_ds.samples:
        past = s["past"]
        y = s["y_abs"]
        preds = pers.predict_np(past, lead_steps)

        # landfall proxy among leads
        t_idx = first_entry_lead_index(y[:, 0], y[:, 1], cfg.florida_bbox)
        p_idx = first_entry_lead_index(preds[:, 0], preds[:, 1], cfg.florida_bbox)

        if t_idx is None and p_idx is None:
            pass
        elif t_idx is None or p_idx is None:
            if cfg.landfall_missing_policy == "max":
                landfall_err_hours.append(float(max(cfg.lead_hours)))
        else:
            landfall_err_hours.append(abs(cfg.lead_hours[p_idx] - cfg.lead_hours[t_idx]))

        for i, h in enumerate(cfg.lead_hours):
            out[f"track_km_{h}h"].append(haversine_km(float(y[i,0]), float(y[i,1]), float(preds[i,0]), float(preds[i,1])))

    metrics = {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}
    metrics["landfall_time_err_hours"] = float(np.mean(landfall_err_hours)) if landfall_err_hours else float("nan")
    # cone metrics are not defined for deterministic persistence
    for h in cfg.lead_hours:
        metrics[f"cone_cov50_{h}h"] = float("nan")
        metrics[f"cone_cov90_{h}h"] = float("nan")
        metrics[f"along_err_km_{h}h"] = float("nan")
        metrics[f"cross_err_km_{h}h"] = float("nan")
    return metrics


# ----------------------------
# Training utilities
# ----------------------------
def train_prob_model(model: nn.Module, tr_loader, te_loader, epochs: int, name: str) -> Dict[str, float]:
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best = float("inf")
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{name}.pt")

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for past, X, meta, y, info in tqdm(tr_loader, desc=f"{name} Ep {ep}/{epochs}", leave=False):
            past = past.to(cfg.device)
            X = X.to(cfg.device)
            meta = meta.to(cfg.device)
            y = y.to(cfg.device)

            mu, sigma = model(past, X, meta)
            loss = gaussian_nll(mu, sigma, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        metrics = evaluate_prob_model(model, te_loader)
        mean_km = float(np.mean([metrics[f"track_km_{h}h"] for h in cfg.lead_hours]))

        print(f"{name} | Ep {ep:02d} | train_nll={np.mean(losses):.4f} | mean_track_km={mean_km:.2f} | landfall_err_h={metrics['landfall_time_err_hours']:.2f}")
        if mean_km < best:
            best = mean_km
            torch.save({"state": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    # load best and return metrics
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["state"])
    final_metrics = evaluate_prob_model(model, te_loader)
    print(f"[{name}] best checkpoint saved:", ckpt_path)
    return final_metrics


def save_metrics_row(model_name: str, metrics: Dict[str, float], out_csv: str):
    row = {"model": model_name, **metrics}
    df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()
    seed_all(cfg.seed)

    # Load data
    irma_df = parse_track(cfg.irma_track, "irma")
    ian_df  = parse_track(cfg.ian_track,  "ian")
    irma_ds = open_era5(cfg.irma_era5)
    ian_ds  = open_era5(cfg.ian_era5)

    print("Building samples (Irma)...")
    s1 = build_samples(irma_df, irma_ds)
    print("Building samples (Ian)...")
    s2 = build_samples(ian_df, ian_ds)

    samples = s1 + s2
    print(f"Total samples: {len(samples)}")
    if len(samples) < 30:
        print("WARNING: few samples. If needed, expand your ERA5 time window and rebuild.")

    idx = np.arange(len(samples))
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=cfg.seed, shuffle=True)

    tr_ds = TrackDataset([samples[i] for i in tr_idx])
    te_ds = TrackDataset([samples[i] for i in te_idx])

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    # ----------------------------
    # Baseline: Persistence (must-have)
    # ----------------------------
    pers_metrics = evaluate_persistence(te_ds)
    save_metrics_row(
        "Persistence",
        pers_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_persistence.csv")
    )

    # ----------------------------
    # Baseline: LSTM
    # ----------------------------
    lstm = LSTMTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    lstm_metrics = train_prob_model(lstm, tr_loader, te_loader, cfg.epochs_baseline, "baseline_lstm")
    save_metrics_row(
        "LSTM (past + ERA5)",
        lstm_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_lstm.csv")
    )

    # ----------------------------
    # Baseline: Transformer
    # ----------------------------
    trm = TransformerTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    trm_metrics = train_prob_model(trm, tr_loader, te_loader, cfg.epochs_baseline, "baseline_transformer")
    save_metrics_row(
        "Transformer (past + ERA5)",
        trm_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_transformer.csv")
    )

    # ----------------------------
    # Primary model: GNO + DynGNN
    # ----------------------------
    main_model = GNO_DynGNN(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    main_metrics = train_prob_model(main_model, tr_loader, te_loader, cfg.epochs_main, "main_gno_dyngnn")
    save_metrics_row(
        "GNO+DynGNN (prob)",
        main_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_gno_dyngnn.csv")
    )

    # ----------------------------
    # Hooks (not executed): HAFS / Pangu / FourCastNet
    # ----------------------------
    print("\n[INFO] HAFS / Pangu / FourCastNet hooks are included as loader stubs.")
    print("       Implement loaders later to benchmark operational + SOTA reference tracks.\n")


if __name__ == "__main__":
    main()
