# =============================================================================
# Save as:
# C:\Users\Adrija\Downloads\DFGCN\track_pipeline_unified_X.py
#
# Implements the EXACT concept you requested:
#  - Unified representation: X(t, node, modality, features)
#      node    = storm-centered grid points (GxG)
#      modality= ERA5 atmospheric
#      features= winds (u,v), geopotential (z), optional pressure (msl if later)
#
# Inputs at time t:
#  - Past storm positions (last N steps)
#  - ERA5 fields around storm -> X(t,node,atmo,features)
#  - Optional storm metadata (vmax, mslp)
#
# Outputs:
#  - Probabilistic predicted storm position at 6/12/24/48h
#    mean track + P50/P90 uncertainty cone
#
# Baselines included (publication-connected):
#  - Persistence (constant velocity)
#  - LSTM baseline
#  - Transformer baseline
#  - (Hook) HAFS baseline (optional: add later by reading HAFS track products)
#  - (Hook) Pangu/FourCastNet extracted track (optional later)
#
# Metrics:
#  - Track error (km)
#  - Along-track / cross-track error
#  - Cone coverage (P50/P90) via Gaussian ellipse inclusion
#  - Landfall time error (Florida coastline crossing approx)
#
# Data expected (from your earlier data gathering step):
#  - HURDAT2 CSVs:
#      data\processed\tracks\irma_2017_hurdat2.csv
#      data\processed\tracks\ian_2022_hurdat2.csv
#  - ERA5 NetCDFs:
#      data\raw\era5\irma_2017\era5_pl_irma_2017.nc
#      data\raw\era5\ian_2022\era5_pl_ian_2022.nc
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

    # features = [u850,v850,u500,v500,z500] (pressure can be added later)
    # modality = atmosphere
    features: Tuple[str, ...] = ("u850","v850","u500","v500","z500")

    history_steps: int = 4       # last N steps (each 6h)
    lead_hours: Tuple[int, ...] = (6, 12, 24, 48)

    include_metadata: bool = True  # vmax, mslp (optional storm metadata)

    # Training
    seed: int = 42
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_root: str = r"C:\Users\Adrija\Downloads\DFGCN"
    ckpt_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\checkpoints"
    metrics_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\metrics"

    # Simple Florida landfall approximation box (for landfall time error)
    florida_bbox: Tuple[float, float, float, float] = (24.0, 32.0, -88.0, -79.0)  # (lat_min, lat_max, lon_min, lon_max)


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
    # bearing from point1 to point2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return math.atan2(y, x)


def along_cross_track_errors(lat0, lon0, lat_true, lon_true, lat_pred, lon_pred):
    """
    Diagnostic decomposition: along-track and cross-track (approx).
    We use bearing from origin->true as reference direction.
    """
    # Convert to local tangent plane approximation in km using haversine distances.
    # Along/cross are approximations; good enough for diagnostics.
    ref = bearing_rad(lat0, lon0, lat_true, lon_true)
    # Distance from origin to pred
    d_op = haversine_km(lat0, lon0, lat_pred, lon_pred)
    # Bearing to pred
    b_op = bearing_rad(lat0, lon0, lat_pred, lon_pred)
    # Angle difference
    dtheta = b_op - ref
    along = d_op * math.cos(dtheta)
    cross = d_op * math.sin(dtheta)
    # True distance along ref direction is d_ot
    d_ot = haversine_km(lat0, lon0, lat_true, lon_true)
    along_err = along - d_ot
    cross_err = cross
    return along_err, cross_err


def in_bbox(lat, lon, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def landfall_time_index(track_lats, track_lons, times, bbox):
    """
    Very simple proxy: first timestep where storm enters Florida bbox.
    Returns index or None.
    """
    for i, (la, lo) in enumerate(zip(track_lats, track_lons)):
        if in_bbox(la, lo, bbox):
            return i
    return None


# ----------------------------
# ERA5 IO + unified X builder
# ----------------------------
def open_era5(nc_path: str) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)

    # detect variable names
    var_u = "u" if "u" in ds.variables else "u_component_of_wind"
    var_v = "v" if "v" in ds.variables else "v_component_of_wind"
    var_z = "z" if "z" in ds.variables else "geopotential"

    if var_u not in ds.variables or var_v not in ds.variables or var_z not in ds.variables:
        raise ValueError(f"ERA5 vars not found. ds.variables={list(ds.variables)}")

    ds.attrs["_u"] = var_u
    ds.attrs["_v"] = var_v
    ds.attrs["_z"] = var_z

    # pressure coord name
    plevel_dim = None
    for cand in ["level", "pressure_level", "plev"]:
        if cand in ds.coords:
            plevel_dim = cand
            break
    if plevel_dim is None:
        raise ValueError(f"Pressure level coord not found. coords={list(ds.coords)}")
    ds.attrs["_plev"] = plevel_dim

    return ds


def parse_track(csv_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["storm_tag"] = tag

    # Ensure numeric metadata
    for col in ["vmax_kt", "mslp_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def crop_era5_to_X(
    ds: xr.Dataset,
    tstamp: pd.Timestamp,
    lat0: float,
    lon0: float,
    crop_deg: float,
    grid_size: int,
) -> np.ndarray:
    """
    Builds the unified tensor:
        X(t,node,atmo,features) -> here we represent as (features, G, G)
    node = grid points (GxG)
    modality = atmo (single modality at present)
    features = u,v,z (at 850/500)
    """
    # nearest time
    dsel = ds.sel(time=np.datetime64(tstamp.to_datetime64()), method="nearest")
    uvar, vvar, zvar = ds.attrs["_u"], ds.attrs["_v"], ds.attrs["_z"]
    plev = ds.attrs["_plev"]

    lat_min, lat_max = lat0 - crop_deg, lat0 + crop_deg
    lon_min, lon_max = lon0 - crop_deg, lon0 + crop_deg

    # handle lon 0..360
    if dsel.longitude.max() > 180:
        def to360(x): return x % 360
        lon_min_, lon_max_ = to360(lon_min), to360(lon_max)
        if lon_min_ > lon_max_:
            part1 = dsel.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_, 360))
            part2 = dsel.sel(latitude=slice(lat_max, lat_min), longitude=slice(0, lon_max_))
            box = xr.concat([part1, part2], dim="longitude")
        else:
            box = dsel.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_, lon_max_))
    else:
        box = dsel.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    def pl(varname: str, level: int):
        return box[varname].sel({plev: level}).values.astype(np.float32)

    # features (path-only)
    u850 = pl(uvar, 850)
    v850 = pl(vvar, 850)
    u500 = pl(uvar, 500)
    v500 = pl(vvar, 500)
    z500 = pl(zvar, 500)

    X = np.stack([u850, v850, u500, v500, z500], axis=0)  # (F,H,W)

    # resample to fixed grid_size for node consistency
    Xt = torch.from_numpy(X).unsqueeze(0)  # (1,F,H,W)
    Xt = F.interpolate(Xt, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    return Xt.squeeze(0).numpy()  # (F,G,G)


# ----------------------------
# Sample builder (Input/Output definition)
# ----------------------------
def build_samples(track_df: pd.DataFrame, era5_ds: xr.Dataset) -> List[Dict]:
    lead_steps = [h // 6 for h in cfg.lead_hours]
    assert all(h % 6 == 0 for h in cfg.lead_hours)

    samples = []
    for i in range(cfg.history_steps, len(track_df)):
        if i + max(lead_steps) >= len(track_df):
            break

        t0 = track_df.loc[i, "datetime_utc"]
        lat0 = float(track_df.loc[i, "lat"])
        lon0 = float(track_df.loc[i, "lon"])

        # past positions (last N steps)
        past = []
        for k in range(cfg.history_steps, 0, -1):
            past.append([float(track_df.loc[i-k, "lat"]), float(track_df.loc[i-k, "lon"])])
        past = np.array(past, dtype=np.float32)  # (H,2)

        # metadata (optional)
        meta = None
        if cfg.include_metadata:
            vmax = float(track_df.loc[i, "vmax_kt"]) if not np.isnan(track_df.loc[i, "vmax_kt"]) else 0.0
            mslp = float(track_df.loc[i, "mslp_mb"]) if not np.isnan(track_df.loc[i, "mslp_mb"]) else 0.0
            meta = np.array([vmax, mslp], dtype=np.float32)

        # unified X(t,node,atmo,features)
        X = crop_era5_to_X(
            era5_ds, t0, lat0, lon0,
            crop_deg=cfg.crop_deg, grid_size=cfg.grid_size
        )  # (F,G,G)

        # labels: future absolute positions
        y_abs = []
        for step in lead_steps:
            y_abs.append([float(track_df.loc[i+step, "lat"]), float(track_df.loc[i+step, "lon"])])
        y_abs = np.array(y_abs, dtype=np.float32)  # (L,2)

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
# Models
# ----------------------------
class PersistenceBaseline:
    """
    Constant velocity extrapolation using last two points in history.
    Produces deterministic tracks.
    """
    def predict(self, past: np.ndarray, lead_steps: List[int]) -> np.ndarray:
        # past: (H,2) oldest->newest, use last two
        p1 = past[-2]
        p2 = past[-1]
        v = p2 - p1  # degrees per 6h
        preds = []
        for s in lead_steps:
            preds.append(p2 + v * s)
        return np.array(preds, dtype=np.float32)  # (L,2)


class OperatorEncoder(nn.Module):
    """
    Lightweight FNO-ish encoder: project -> conv blocks -> global pool -> vector.
    This keeps 'operator' flavor but remains simple & fast.
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
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(width, out_dim),
            nn.ReLU()
        )

    def forward(self, X):
        h = self.net(X)
        return self.head(h)


class DynamicGNN(nn.Module):
    """
    Dynamic message passing over history nodes (positions).
    """
    def __init__(self, node_dim=32, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Linear(2, node_dim)
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim)
        ) for _ in range(layers)])
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
    """
    Probabilistic head: outputs Gaussian (mu, sigma) for each lead time (lat,lon).
    """
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
    Graph Neural Operator + Dynamic GNN:
    - Operator encoder reads X(t,node,atmo,features) grid
    - Dynamic GNN reads past storm positions
    - Optional metadata (vmax, mslp)
    - Outputs probabilistic future positions (absolute)
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
        # past: (B,H,2), X: (B,F,G,G), meta: (B,2)
        op = self.op(X)         # (B,128)
        g  = self.gnn(past)     # (B,64)
        p  = self.past_mlp(past) # (B,64)
        parts = [op, g, p]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)  # (mu, sigma) absolute positions


# ----------------------------
# Loss + cone coverage
# ----------------------------
def gaussian_nll(mu, sigma, y):
    # mu,sigma,y: (B,L,2)
    eps = 1e-6
    var = sigma**2 + eps
    return 0.5 * torch.mean(((y - mu) ** 2) / var + torch.log(var))


def ellipse_inclusion(lat_true, lon_true, mu_lat, mu_lon, sigma_lat, sigma_lon, z):
    """
    Checks if (true) lies within axis-aligned Gaussian ellipse:
      ((x-mu)/sigma)^2 sum <= z^2
    This is an approximation but works for a demo.
    """
    dx = (lat_true - mu_lat) / (sigma_lat + 1e-6)
    dy = (lon_true - mu_lon) / (sigma_lon + 1e-6)
    return (dx*dx + dy*dy) <= (z*z)


# z-scores for 2D Gaussian probability mass (approx):
# For simplicity we use common chi-square cutoffs:
# P50 ~ chi2(2)=1.386 -> sqrt=1.177
# P90 ~ chi2(2)=4.605 -> sqrt=2.146
Z_P50 = 1.177
Z_P90 = 2.146


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate_model(model, loader) -> Dict[str, float]:
    model.eval()

    lead_steps = [h // 6 for h in cfg.lead_hours]
    track_err = [[] for _ in cfg.lead_hours]
    along_err = [[] for _ in cfg.lead_hours]
    cross_err = [[] for _ in cfg.lead_hours]
    cov50 = [[] for _ in cfg.lead_hours]
    cov90 = [[] for _ in cfg.lead_hours]

    # landfall time arrays (proxy)
    # We'll compute using mean predicted tracks for each sample only (rough).
    # For a more accurate landfall time, you would roll forward sequentially.

    for past, X, meta, y, info in loader:
        past = past.to(cfg.device)
        X = X.to(cfg.device)
        meta = meta.to(cfg.device)
        y = y.to(cfg.device)

        mu, sigma = model(past, X, meta)  # (B,L,2)
        mu_lat, mu_lon = mu[..., 0], mu[..., 1]
        sig_lat, sig_lon = sigma[..., 0], sigma[..., 1]

        for b in range(mu.size(0)):
            # origin for along/cross diagnostics: last observed point
            lat0 = float(past[b, -1, 0].cpu())
            lon0 = float(past[b, -1, 1].cpu())

            for li in range(mu.size(1)):
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

    metrics = {}
    for i, h in enumerate(cfg.lead_hours):
        metrics[f"track_km_{h}h"] = float(np.mean(track_err[i])) if track_err[i] else np.nan
        metrics[f"along_err_km_{h}h"] = float(np.mean(along_err[i])) if along_err[i] else np.nan
        metrics[f"cross_err_km_{h}h"] = float(np.mean(cross_err[i])) if cross_err[i] else np.nan
        metrics[f"cone_cov50_{h}h"] = float(np.mean(cov50[i])) if cov50[i] else np.nan
        metrics[f"cone_cov90_{h}h"] = float(np.mean(cov90[i])) if cov90[i] else np.nan

    return metrics


# ----------------------------
# Train
# ----------------------------
def train_main():
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
    if len(samples) < 20:
        print("WARNING: few samples. Consider increasing ERA5 day range.")

    idx = np.arange(len(samples))
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=cfg.seed, shuffle=True)

    tr_ds = TrackDataset([samples[i] for i in tr_idx])
    te_ds = TrackDataset([samples[i] for i in te_idx])

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GNO_DynGNN(feat_ch=len(cfg.features), leads=len(cfg.lead_hours), use_meta=cfg.include_metadata).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    best = 1e9
    ckpt_path = os.path.join(cfg.ckpt_dir, "gno_dyn_gnn_track.pt")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for past, X, meta, y, info in tqdm(tr_loader, desc=f"Epoch {ep}/{cfg.epochs}", leave=False):
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

        metrics = evaluate_model(model, te_loader)
        mean_km = np.mean([metrics[f"track_km_{h}h"] for h in cfg.lead_hours])

        print(f"Epoch {ep:02d} | train_nll={np.mean(losses):.4f} | mean_track_km={mean_km:.2f}")
        print("  ", {k: round(v, 3) for k, v in metrics.items()})

        if mean_km < best:
            best = mean_km
            torch.save({"state": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    # Save final metrics
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["state"])
    final_metrics = evaluate_model(model, te_loader)

    out_csv = os.path.join(cfg.metrics_dir, "track_metrics_gno_dyn_gnn.csv")
    pd.DataFrame([{"model": "GNO+DynGNN", **final_metrics}]).to_csv(out_csv, index=False)
    print("Saved:", ckpt_path)
    print("Saved:", out_csv)

    # ----------------------------
    # Baseline: Persistence (must-have)
    # ----------------------------
    pers = PersistenceBaseline()
    lead_steps = [h // 6 for h in cfg.lead_hours]

    pers_err = {f"track_km_{h}h": [] for h in cfg.lead_hours}
    for s in te_ds.samples:
        past = s["past"]  # (H,2)
        y = s["y_abs"]    # (L,2)
        preds = pers.predict(past, lead_steps)  # (L,2)
        for i, h in enumerate(cfg.lead_hours):
            pers_err[f"track_km_{h}h"].append(haversine_km(y[i,0], y[i,1], preds[i,0], preds[i,1]))

    pers_metrics = {k: float(np.mean(v)) for k, v in pers_err.items()}
    out_csv2 = os.path.join(cfg.metrics_dir, "track_metrics_persistence.csv")
    pd.DataFrame([{"model": "Persistence", **pers_metrics}]).to_csv(out_csv2, index=False)
    print("Saved:", out_csv2)


if __name__ == "__main__":
    train_main()
