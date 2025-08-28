#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch pipeline for training an MLP on merged text+image embeddings from Parquet.

- Все пути/гиперпараметры задаются через dataclasses (Paths, RunCfg)
- Нормализация типа ключа id (auto|str|int) перед merge и выравниванием таргета
- Стандартизация фич (по train) и применение к val/test
- BCEWithLogitsLoss + pos_weight при дисбалансе
- AMP (новый API torch.amp) и early stopping по val_loss
- Визуализация 2D (UMAP при наличии umap-learn, иначе PCA)

Зависимости:
  pip install pandas pyarrow torch scikit-learn umap-learn matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# --- dimensionality reduction import (robust) ---
try:
    import umap.umap_ as umap
    _HAS_UMAP = True
except Exception:
    from sklearn.decomposition import PCA
    _HAS_UMAP = False


# =========================
# Config dataclasses
# =========================
@dataclass
class Paths:
    train_image_parquet: str
    train_text_parquet: str
    train_dataset_csv: Optional[str] = None  # must contain id + target
    test_image_parquet: Optional[str] = None
    test_text_parquet: Optional[str] = None


@dataclass
class RunCfg:
    id_col: Optional[str]                  # e.g., "id"; if None, index-based alignment
    target_col: Optional[str]              # e.g., "resolution"
    test_size: float
    random_state: int
    batch_size: int
    lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    hidden_sizes: List[int]
    dropout: float
    standardize: bool = True
    use_amp: bool = True                   # enable AMP if supported
    num_workers: int = 2
    pin_memory: bool = True
    id_dtype: str = "auto"                 # "auto" | "str" | "int"


# =========================
# Helpers for ID dtype
# =========================
def _unify_merge_key(
    left: pd.DataFrame, right: pd.DataFrame, id_col: str, mode: str = "auto"
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Ensure left[id_col] and right[id_col] have the same dtype suitable for merge.
    Returns (left,right, resolved_mode) where resolved_mode in {"str","int"}.
    """
    if mode not in {"auto", "str", "int"}:
        raise ValueError("id_dtype must be one of {'auto','str','int'}")

    if mode == "str":
        left[id_col] = left[id_col].astype(str)
        right[id_col] = right[id_col].astype(str)
        return left, right, "str"

    if mode == "int":
        left[id_col]  = pd.to_numeric(left[id_col], errors="raise").astype("int64")
        right[id_col] = pd.to_numeric(right[id_col], errors="raise").astype("int64")
        return left, right, "int"

    # auto
    if left[id_col].dtype == right[id_col].dtype and pd.api.types.is_integer_dtype(left[id_col].dtype):
        left[id_col]  = left[id_col].astype("int64")
        right[id_col] = right[id_col].astype("int64")
        return left, right, "int"

    l_num = pd.to_numeric(left[id_col], errors="coerce")
    r_num = pd.to_numeric(right[id_col], errors="coerce")
    if l_num.notna().all() and r_num.notna().all():
        left[id_col]  = l_num.astype("int64")
        right[id_col] = r_num.astype("int64")
        return left, right, "int"

    left[id_col]  = left[id_col].astype(str)
    right[id_col] = right[id_col].astype(str)
    return left, right, "str"


def _coerce_series_index_dtype(y: pd.Series, id_col: str, resolved_mode: str) -> pd.Series:
    """Coerce y.index (ids) to the same dtype chosen for merge."""
    if y.index.name != id_col:
        if id_col in y.index.names:
            y = y.reset_index().set_index(id_col)[y.name]
        else:
            raise KeyError(f"Labels Series must be indexed by '{id_col}'.")
    if resolved_mode == "int":
        y.index = pd.to_numeric(y.index, errors="raise").astype("int64")
    else:
        y.index = y.index.astype(str)
    return y


# =========================
# Dataset
# =========================
class TextImageData(Dataset):
    def __init__(
        self,
        img_path: str,
        txt_path: str,
        y: Optional[pd.Series] = None,
        id_col: str = "id",
        id_dtype_mode: str = "auto",
    ):
        img_df = pd.read_parquet(img_path, engine='pyarrow')
        txt_df = pd.read_parquet(txt_path, engine='pyarrow')

        if id_col not in img_df.columns or id_col not in txt_df.columns:
            raise KeyError(f"Column '{id_col}' must exist in both parquet files.")

        img_df, txt_df, resolved = _unify_merge_key(img_df.copy(), txt_df.copy(), id_col, id_dtype_mode)

        if y is not None:
            y = _coerce_series_index_dtype(y.copy(), id_col, resolved)

        whole_df = pd.merge(txt_df, img_df, on=id_col, how='inner')
        self.ids = whole_df[id_col].values

        if y is not None:
            aligned = y.reindex(whole_df[id_col].values)
            if aligned.isna().any():
                missing = int(aligned.isna().sum())
                raise ValueError(f"{missing} labels are missing after alignment by '{id_col}'.")
            self.y = torch.tensor(aligned.values, dtype=torch.float32)
        else:
            self.y = None

        self.whole_data = torch.tensor(whole_df.drop(columns=[id_col]).values, dtype=torch.float32)

        # free memory
        del img_df, txt_df, whole_df
        gc.collect()

    def __len__(self):
        return self.whole_data.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.whole_data[idx], self.y[idx]
        else:
            return self.whole_data[idx]

    def get_pos_weights(self):
        if self.y is None:
            return None
        pos = self.y.sum()
        neg = self.y.shape[0] - pos
        if pos.item() == 0:
            return torch.tensor(1.0, dtype=torch.float32)
        return (neg / pos).to(dtype=torch.float32)

    def get_input_dim(self):
        return self.whole_data.shape[1]


# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last = h
        self.feature = nn.Sequential(*layers)     # feature extractor
        self.head = nn.Linear(last, 1)            # binary head
        self.net = nn.Sequential(self.feature, self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.feature(x)  # [B, hidden_last]


# =========================
# Training utils
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler_amp: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: Optional[float] = 2.0,
):
    model.train()
    total_loss = 0.0
    n_samples = 0
    use_autocast = scaler_amp is not None
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            y = None

        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, y) if (y is not None) else (logits ** 2).mean()
            scaler_amp.scale(loss).backward()
            if max_grad_norm:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y) if (y is not None) else (logits ** 2).mean()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)

    metrics = {
        "val_loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, probs))
    except Exception:
        metrics["avg_precision"] = float("nan")

    return metrics


def compute_standardization_stats(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def apply_standardization_inplace(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    tensor.sub_(mean).div_(std)


def make_embedding_plot(X: np.ndarray, y: Optional[np.ndarray], title: str, out_path: Path):
    """
    2D plot via UMAP if available, else PCA.
    """
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb2d = reducer.fit_transform(X)
    else:
        print(f"[UMAP] umap-learn not found, falling back to PCA -> {out_path}")
        reducer = PCA(n_components=2, random_state=42)
        emb2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if y is not None:
        for cls in np.unique(y):
            mask = (y == cls)
            plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=8, alpha=0.85, label=str(cls))
        plt.legend(markerscale=2, fontsize='small')
    else:
        plt.scatter(emb2d[:, 0], emb2d[:, 1], s=8, alpha=0.85)
    plt.title(title)
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


# =========================
# Orchestration
# =========================
def run_training(paths: Paths, cfg: RunCfg, out_dir: str = "./artifacts_torch") -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = cfg.use_amp and (device.type in ("cuda", "mps"))

    # ---- Load labels (train) ----
    y_series = None
    if cfg.target_col and paths.train_dataset_csv:
        meta = pd.read_csv(paths.train_dataset_csv)
        if cfg.id_col not in meta.columns or cfg.target_col not in meta.columns:
            raise KeyError(f"train_dataset_csv must contain columns '{cfg.id_col}' and '{cfg.target_col}'.")
        y_series = meta.set_index(cfg.id_col)[cfg.target_col].astype(float)
        y_series.index.name = cfg.id_col

    # ---- Build full train dataset ----
    ds_full = TextImageData(
        paths.train_image_parquet,
        paths.train_text_parquet,
        y=y_series,
        id_col=cfg.id_col or "id",
        id_dtype_mode=cfg.id_dtype,
    )

    # ---- Standardization (from train only) ----
    mean = std = None
    if cfg.standardize:
        mean, std = compute_standardization_stats(ds_full.whole_data)
        apply_standardization_inplace(ds_full.whole_data, mean, std)

    # ---- Train/Val split indices ----
    n = len(ds_full)
    rng = np.random.RandomState(cfg.random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = int(np.floor(cfg.test_size * n))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    ds_train = Subset(ds_full, train_idx.tolist())
    ds_val = Subset(ds_full, val_idx.tolist())

    # ---- DataLoaders ----
    loader_args = dict(batch_size=cfg.batch_size,
                       num_workers=cfg.num_workers,
                       pin_memory=(cfg.pin_memory and device.type == "cuda"))
    train_loader = DataLoader(ds_train, shuffle=True, **loader_args)
    val_loader = DataLoader(ds_val, shuffle=False, **loader_args)

    # ---- Model ----
    in_dim = ds_full.get_input_dim()
    model = MLP(in_dim, cfg.hidden_sizes, cfg.dropout).to(device)

    # ---- Loss with pos_weight ----
    if ds_full.y is None:
        raise ValueError("Supervised training requires target labels. Provide target_col and train_dataset_csv.")
    pos_w = ds_full.get_pos_weights()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- AMP scaler (new API) ----
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    # ---- Training loop with early stopping ----
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    start_time = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                     device, scaler_amp=scaler, max_grad_norm=2.0)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["val_loss"] < best_val - 1e-6:
            best_val = val_metrics["val_loss"]
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['val_loss']:.4f} | acc={val_metrics.get('accuracy', float('nan')):.4f} "
              f"| f1={val_metrics.get('f1_macro', float('nan')):.4f}")

        if no_improve >= cfg.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - start_time

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # ---- Save artifacts ----
    (out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save({"optimizer": optimizer.state_dict()}, out_dir / "optimizer.pt")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"paths": asdict(paths), "cfg": asdict(cfg), "device": device.type}, f, ensure_ascii=False, indent=2)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ---- 2D plots (inputs / features) ----
    X_np = ds_full.whole_data.cpu().numpy()
    y_np = ds_full.y.cpu().numpy() if ds_full.y is not None else None
    make_embedding_plot(X_np, y_np, title="2D projection of input embeddings", out_path=out_dir / "proj_input.png")

    model.eval()
    feats = []
    with torch.no_grad():
        for batch in DataLoader(ds_full, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z = model.extract_features(x)
            feats.append(z.cpu())
    feats_np = torch.cat(feats, dim=0).numpy()
    make_embedding_plot(feats_np, y_np, title="2D projection of learned features", out_path=out_dir / "proj_features.png")

    # Save standardization stats
    if cfg.standardize and (mean is not None and std is not None):
        torch.save({"mean": mean.cpu(), "std": std.cpu()}, out_dir / "standardization.pt")

    # ---- Optional: test set inference ----
    test_summary = {}
    if paths.test_image_parquet and paths.test_text_parquet:
        ds_test = TextImageData(
            paths.test_image_parquet,
            paths.test_text_parquet,
            y=None,
            id_col=cfg.id_col or "id",
            id_dtype_mode=cfg.id_dtype,
        )
        if cfg.standardize and (mean is not None and std is not None):
            apply_standardization_inplace(ds_test.whole_data, mean, std)

        test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        model.eval()
        all_logits = []
        with torch.no_grad():
            for xb in test_loader:
                xb = xb[0] if isinstance(xb, (list, tuple)) else xb
                xb = xb.to(device)
                lg = model(xb)
                all_logits.append(lg.cpu())
        logits = torch.cat(all_logits).numpy()
        probs = 1 / (1 + np.exp(-logits))
        pred_df = pd.DataFrame({"id": ds_test.ids, "logit": logits.flatten(), "prob": probs.flatten()})
        pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
        test_summary["num_test_rows"] = len(ds_test)

    # ---- Final summary ----
    summary = {
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "total_time_sec": total_time,
        **({"test": test_summary} if test_summary else {})
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    paths = Paths(
        train_image_parquet="data/train_siglip2_embeddings-1.parquet",
        train_text_parquet ="data/train_text_features.parquet",
        train_dataset_csv   ='data/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv',
        test_image_parquet ="data/test_siglip2_embeddings.parquet",
        test_text_parquet  ="data/test_text_features.parquet",
    )

    cfg = RunCfg(
        id_col="id",
        target_col="resolution",
        test_size=0.2,
        random_state=42,
        batch_size=64,
        lr=2e-3,
        weight_decay=1e-4,
        max_epochs=50,
        patience=5,
        hidden_sizes=[1024, 512, 128],
        dropout=0.2,
        standardize=True,
        id_dtype="auto",   # можно задать "str" или "int", если тип ключа известен заранее
    )

    out = run_training(paths, cfg, out_dir="./artifacts_torch")
    print("Done. Summary:", out)
