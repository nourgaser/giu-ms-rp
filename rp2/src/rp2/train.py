"""
train.py — cross-validated training, identical for every model.

One loop, used by all three models. That is the point: if the baseline and
Tri-CAN were trained by different code with different early-stopping rules,
the comparison would be between training recipes rather than architectures.

Design decisions and why:

  * Stratified K-fold, not a single split. EGY-MER has 363 samples, where a
    single test split is a coin toss; the same protocol is used on Ar-MuSA so
    that the two are reported the same way.
  * The SAME folds for every model, generated from a fixed seed, so every
    comparison is paired.
  * Class-weighted cross-entropy by default. Ar-MuSA is mildly imbalanced
    (neutral is 22.5%); EGY-MER severely so (fear and surprise under 4%).
    Focal loss is available as an option, but weighted CE is the honest
    default because it has one fewer knob to tune.
  * Early stopping on validation macro-F1 with a fixed patience, restoring the
    best weights. Applied identically to every model.
  * Seeds are set per (fold, model, repeat) and recorded, so any single number
    in the report can be reproduced exactly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from . import models as M


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.3
    loss: str = "weighted_ce"   # or "focal"
    focal_gamma: float = 2.0
    n_folds: int = 5
    seed: int = 0
    n_threads: int = 0          # 0 = leave torch's default


class FeatureDataset(Dataset):
    """Wraps the cached feature arrays. Features are frozen, so this is just
    indexing — no augmentation, no encoder forward pass."""

    def __init__(self, feats: dict[str, np.ndarray], labels: np.ndarray,
                 masks: dict[str, np.ndarray] | None = None) -> None:
        self.mods = tuple(sorted(feats))
        self.feats = feats
        self.masks = masks or {}
        self.labels = labels
        n = len(labels)
        for m in self.mods:
            if len(feats[m]) != n:
                raise ValueError(f"modality {m!r}: {len(feats[m])} rows vs {n} labels")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = {m: torch.from_numpy(np.asarray(self.feats[m][i])) for m in self.mods}
        msk = {m: torch.from_numpy(np.asarray(self.masks[m][i]))
               for m in self.mods if m in self.masks}
        return x, msk, int(self.labels[i])


def _collate(batch):
    xs, ms, ys = zip(*batch)
    mods = xs[0].keys()
    feats = {m: torch.stack([x[m] for x in xs]) for m in mods}
    masks = {m: torch.stack([x[m] for x in ms]) for m in mods if m in ms[0]}
    return feats, masks, torch.tensor(ys, dtype=torch.long)


class FocalLoss(nn.Module):
    """Focal loss, offered because the proposal specified it.

    Kept optional rather than default: on a 3-class corpus with mild imbalance
    it mostly adds a hyperparameter. On EGY-MER, where two classes are under
    4%, it may earn its place — that is an empirical question for when the
    corpus arrives, not something to assume now.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.numel() else None
        ce = nn.functional.cross_entropy(logits, target, weight=w, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _make_loss(cfg: TrainConfig, y: np.ndarray, n_labels: int) -> nn.Module:
    counts = np.bincount(y, minlength=n_labels).astype(np.float64)
    # Inverse-frequency weights, normalised to mean 1 so the effective learning
    # rate does not shift with the number of classes.
    w = np.where(counts > 0, counts.sum() / (n_labels * np.maximum(counts, 1)), 0.0)
    w = w / w[w > 0].mean() if (w > 0).any() else w
    wt = torch.tensor(w, dtype=torch.float32)
    if cfg.loss == "focal":
        return FocalLoss(weight=wt, gamma=cfg.focal_gamma)
    return nn.CrossEntropyLoss(weight=wt)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_fold(
    model: nn.Module,
    train_ds: FeatureDataset,
    val_ds: FeatureDataset,
    cfg: TrainConfig,
    n_labels: int,
    seed: int,
) -> tuple[nn.Module, dict]:
    """Train to best validation macro-F1, restore best weights, return them."""
    _seed_everything(seed)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=_collate, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=_collate)

    crit = _make_loss(cfg, train_ds.labels, n_labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for feats, masks, y in train_dl:
            opt.zero_grad()
            loss = crit(model(feats, masks), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(y)
        sched.step()

        yt, yp = predict(model, val_dl)
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total / max(len(train_ds), 1),
                        "val_macro_f1": f1})

        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_macro_f1": best_f1, "best_epoch": best_epoch,
                   "epochs_run": len(history), "history": history}


@torch.no_grad()
def predict(model: nn.Module, dl: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    yt, yp = [], []
    for feats, masks, y in dl:
        logits = model(feats, masks)
        yp.append(logits.argmax(dim=-1).numpy())
        yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def cross_validate(
    kind: str,
    feats: dict[str, np.ndarray],
    labels: np.ndarray,
    n_labels: int,
    cfg: TrainConfig,
    masks: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run stratified K-fold for one model. Returns out-of-fold predictions.

    Every sample is predicted exactly once, by a model that never saw it in
    training, so the returned arrays can be sliced by congruence stratum after
    the fact without any leakage.
    """
    if cfg.n_threads:
        torch.set_num_threads(cfg.n_threads)

    dims = {m: int(np.asarray(feats[m]).shape[-1]) for m in feats}
    n = len(labels)
    oof_pred = np.full(n, -1, dtype=int)
    fold_id = np.full(n, -1, dtype=int)
    fold_info = []

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    for k, (tr, va) in enumerate(skf.split(np.zeros(n), labels)):
        seed = cfg.seed * 1000 + k
        _seed_everything(seed)

        # Only the attention-based models take n_heads; passing it to the
        # concatenation baselines would be a TypeError.
        needs_heads = (kind.startswith(("tri-can", "hybrid"))
                       or kind in {"all-pairs", "early-fusion"})
        model = M.build(kind, dims, n_labels, d_model=cfg.d_model,
                        dropout=cfg.dropout,
                        **({"n_heads": cfg.n_heads} if needs_heads else {}))

        sub = lambda d, idx: {m: d[m][idx] for m in d} if d else None
        train_ds = FeatureDataset(sub(feats, tr), labels[tr], sub(masks, tr))
        val_ds = FeatureDataset(sub(feats, va), labels[va], sub(masks, va))

        model, info = train_one_fold(model, train_ds, val_ds, cfg, n_labels, seed)

        _, yp = predict(model, DataLoader(val_ds, batch_size=256,
                                          shuffle=False, collate_fn=_collate))
        oof_pred[va] = yp
        fold_id[va] = k
        info.update({"fold": k, "seed": seed, "n_train": len(tr), "n_val": len(va),
                     "params": M.count_params(model)})
        info.pop("history")
        fold_info.append(info)

    if (oof_pred < 0).any():
        raise RuntimeError("some samples received no out-of-fold prediction")

    return {"model": kind, "y_pred": oof_pred, "fold": fold_id, "folds": fold_info,
            "params": fold_info[0]["params"] if fold_info else 0}
