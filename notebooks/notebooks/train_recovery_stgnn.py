# # -*- coding: utf-8 -*-
r"""
train_recovery_stgnn.py

Minimal, dependency-light training script for a recovery prediction baseline.

Loads graph dataset artifacts (already generated in recovery_dataset_out):
  - X_static.npy      (N, F_static)
  - X_dynamic.npy     (T, N, F_dyn, E)  OR (T, N, F_dyn)   (E = ensemble members)
  - edge_index.npy    (2, M)   int64
  - edge_weight.npy   (M,)     float32
  - nodes.csv         should contain a stable node id column (e.g., FIPS/cell_id) + lat/lon
  - node_ids.json     mapping: node_id (string) -> node index (0..N-1)
  - Y.npy (optional)  (T, N) recovery target in [0,1]

Model:
  - GraphConv (weighted, row-normalized sparse adjacency) on static features
  - GRU over time using [graph_embedding + dynamic_features]
  - MLP head -> recovery in [0,1]

Outputs:
  - pred_recovery.npy  (T, N)
  - pred_recovery.csv  columns: t,node_id,pred_recovery   (if --save_csv)

Usage (PowerShell):
  python .\train_recovery_stgnn.py --data_dir "C:\Users\Adrija\Downloads\DFGCN\recovery_dataset_out" --epochs 5 --save_csv

Notes:
  - If Y.npy is missing, the script skips training and runs forward inference only.
  - For ensembles in X_dynamic.npy, the script uses mean over E by default and can optionally save std.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def load_arrays(data_dir: Path):
    # Required
    X_static = np.load(data_dir / "X_static.npy")          # (N, F_static)
    edge_index = np.load(data_dir / "edge_index.npy")      # (2, M)
    edge_weight = np.load(data_dir / "edge_weight.npy")    # (M,)
    X_dynamic = np.load(data_dir / "X_dynamic.npy")        # (T, N, F_dyn, E) or (T, N, F_dyn)

    # Optional labels
    y_path = data_dir / "Y.npy"
    Y = np.load(y_path) if y_path.exists() else None

    # Node id mapping
    with open(data_dir / "node_ids.json", "r", encoding="utf-8") as f:
        node_to_idx = json.load(f)

    # Invert mapping for output
    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}
    return X_static, X_dynamic, edge_index, edge_weight, Y, idx_to_node


def build_sparse_adj(edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int, device: torch.device):
    """
    Build row-normalized weighted adjacency A_hat suitable for sparse mm:
        A_hat = D^{-1} A
    Where A aggregates from src -> dst.
    """
    src = torch.from_numpy(edge_index[0]).long().to(device)
    dst = torch.from_numpy(edge_index[1]).long().to(device)
    w = torch.from_numpy(edge_weight).float().to(device)

    # A has entries at (dst, src) so mm(A, X) aggregates src -> dst
    indices = torch.stack([dst, src], dim=0)
    A = torch.sparse_coo_tensor(indices, w, size=(num_nodes, num_nodes), device=device).coalesce()

    # Row sums
    row_sum = torch.sparse.sum(A, dim=1).to_dense()  # (N,)
    inv = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))

    # Normalize values by destination row (indices[0] == dst)
    r = indices[0]
    A_hat = torch.sparse_coo_tensor(indices, w * inv[r], size=(num_nodes, num_nodes), device=device).coalesce()
    return A_hat


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor):
        # X: (N, Fin)
        Xw = self.lin(X)                 # (N, Fout)
        out = torch.sparse.mm(A_hat, Xw) # (N, Fout)
        return out


class STGNN(nn.Module):
    """
    Baseline:
      - GraphConv on static features -> node embedding
      - For each time t, concat [node_embedding, dyn_features_mean]
      - GRU over time per node
      - MLP head -> recovery in [0,1]
    """
    def __init__(self, f_static: int, f_dyn: int, g_hidden: int = 32, r_hidden: int = 64):
        super().__init__()
        self.g1 = GraphConv(f_static, g_hidden)
        self.g2 = GraphConv(g_hidden, g_hidden)
        self.act = nn.ReLU()

        self.gru = nn.GRU(input_size=g_hidden + f_dyn, hidden_size=r_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(r_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X_static: torch.Tensor, X_dyn_mean: torch.Tensor, A_hat: torch.Tensor):
        """
        X_static:   (N, F_static)
        X_dyn_mean: (T, N, F_dyn)
        Returns:
          pred: (T, N)
        """
        # Graph embedding
        h = self.act(self.g1(X_static, A_hat))
        h = self.act(self.g2(h, A_hat))  # (N, g_hidden)

        T, N, Fd = X_dyn_mean.shape

        # (T,N,g_hidden)
        h_rep = h.unsqueeze(0).expand(T, N, h.shape[-1])
        # (T,N,g_hidden+Fd)
        z = torch.cat([h_rep, X_dyn_mean], dim=-1)

        # GRU expects (N,T,feat)
        z = z.permute(1, 0, 2).contiguous()

        out, _ = self.gru(z)             # (N,T,r_hidden)
        y = self.head(out).squeeze(-1)   # (N,T)
        y = y.permute(1, 0).contiguous() # (T,N)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing X_static.npy, X_dynamic.npy, edge_index.npy, edge_weight.npy, node_ids.json, (optional) Y.npy")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--save_csv", action="store_true", help="Also save pred_recovery.csv for dashboard.")
    ap.add_argument("--save_unc_std", action="store_true", help="If ensembles exist, save weather_uncertainty_std.npy.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    X_static_np, X_dynamic_np, edge_index_np, edge_weight_np, Y_np, idx_to_node = load_arrays(data_dir)

    # X_dynamic: (T,N,F,E) -> mean over E, or already (T,N,F)
    if X_dynamic_np.ndim == 4:
        X_dyn_mean_np = X_dynamic_np.mean(axis=-1)  # (T,N,F)
        X_dyn_std_np = X_dynamic_np.std(axis=-1)    # (T,N,F)
    elif X_dynamic_np.ndim == 3:
        X_dyn_mean_np = X_dynamic_np
        X_dyn_std_np = None
    else:
        raise ValueError(f"Unexpected X_dynamic shape: {X_dynamic_np.shape}. Expected 3D or 4D.")

    T, N, F_dyn = X_dyn_mean_np.shape
    if X_static_np.shape[0] != N:
        raise ValueError(f"Node mismatch: X_static has N={X_static_np.shape[0]} but X_dynamic has N={N}")

    device = torch.device(args.device)
    X_static = torch.from_numpy(X_static_np).float().to(device)
    X_dyn_mean = torch.from_numpy(X_dyn_mean_np).float().to(device)

    A_hat = build_sparse_adj(edge_index_np, edge_weight_np, num_nodes=N, device=device)

    model = STGNN(f_static=X_static.shape[1], f_dyn=F_dyn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Labels (optional)
    if Y_np is not None:
        Y = torch.from_numpy(Y_np).float().to(device)
        if Y.ndim != 2:
            raise ValueError(f"Unexpected Y shape: {Y.shape}. Expected (T,N).")

        T_y, N_y = Y.shape
        if N_y != N:
            raise ValueError(f"Y node mismatch: Y has N={N_y}, expected N={N}")
        T_use = min(T, T_y)
        X_dyn_mean = X_dyn_mean[:T_use]
        Y = Y[:T_use]
        T = T_use
    else:
        Y = None

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] N={N}, T={T}, F_static={X_static.shape[1]}, F_dyn={F_dyn}, device={device}")
    print(f"[INFO] Labels: {'FOUND (training enabled)' if Y is not None else 'NOT FOUND (inference only)'}")

    # Train if labels exist
    if Y is not None:
        model.train()
        for ep in range(1, args.epochs + 1):
            opt.zero_grad()
            pred = model(X_static, X_dyn_mean, A_hat)  # (T,N)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
            if ep == 1 or ep % max(1, args.epochs // 5) == 0:
                print(f"epoch {ep:03d}/{args.epochs}  loss={loss.item():.6f}")

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(X_static, X_dyn_mean, A_hat).detach().cpu().numpy()  # (T,N)

    out_npy = data_dir / "pred_recovery.npy"
    np.save(out_npy, pred.astype(np.float32))
    print("[OK] Saved:", out_npy, pred.shape)

    if args.save_csv:
        rows = [(t, idx_to_node.get(i, str(i)), float(pred[t, i]))
                for t in range(pred.shape[0])
                for i in range(pred.shape[1])]
        out_csv = data_dir / "pred_recovery.csv"
        pd.DataFrame(rows, columns=["t", "node_id", "pred_recovery"]).to_csv(out_csv, index=False)
        print("[OK] Saved:", out_csv)

    if args.save_unc_std and (X_dyn_std_np is not None):
        unc_out = data_dir / "weather_uncertainty_std.npy"
        np.save(unc_out, X_dyn_std_np.astype(np.float32))
        print("[OK] Saved:", unc_out, X_dyn_std_np.shape)


if __name__ == "__main__":
    main()
