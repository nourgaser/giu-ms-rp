"""
finetune.py — the same fusion mechanisms, with the text encoder unfrozen.

Why this exists. Every result so far uses frozen encoders, which was forced by
the hardware and defended as a clean ablation: with the features fixed, the only
difference between late fusion and Tri-CAN is the fusion mechanism. That is
true, but it leaves a real question open. Our text-only macro-F1 is ~0.63 while
published Ar-MuSA results reach 0.78-0.90, and those papers fine-tune. If
cross-modal attention only pays off once the representations are good enough,
a frozen-feature study would never see it, and the null would be an artefact of
the protocol rather than a fact about the mechanism.

So: same fusion modules, same folds, same metrics — text encoder trainable.

Only the TEXT encoder is unfrozen, for two reasons. It carries almost all of
the signal (text-only 0.63 against 0.42 audio and 0.43 visual), and it is the
only one of the three that is cheap enough to backprop through on a CPU:
AraBERT over 48 tokens is roughly 45 ms/sample forward+backward, whereas
Wav2Vec2 over 8 seconds of audio is orders of magnitude worse. Audio and visual
features stay cached and frozen. That is stated as a limitation rather than
presented as a design choice.

Two parameter groups, because one learning rate cannot serve both: a pretrained
encoder needs ~2e-5 or it forgets what it knows, while a randomly-initialised
fusion block and head need ~1e-3 to learn anything in three epochs.
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
from .train import _make_loss, _seed_everything


@dataclass
class FinetuneConfig:
    encoder: str = "aubmindlab/bert-base-arabertv02-twitter"
    epochs: int = 3
    batch_size: int = 16
    lr_encoder: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    max_len: int = 48
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.3
    n_folds: int = 3
    seed: int = 0
    n_threads: int = 11
    warmup_frac: float = 0.1


class TokenizedDataset(Dataset):
    """Pre-tokenised text plus the cached (frozen) audio/visual features."""

    def __init__(self, enc: dict, feats: dict[str, np.ndarray],
                 masks: dict[str, np.ndarray], labels: np.ndarray) -> None:
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.feats = feats
        self.masks = masks
        self.labels = labels
        self.other = tuple(m for m in sorted(feats) if m != "text")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = {m: torch.from_numpy(np.asarray(self.feats[m][i])) for m in self.other}
        k = {m: torch.from_numpy(np.asarray(self.masks[m][i]))
             for m in self.other if m in self.masks}
        return (self.ids[i], self.attn[i], x, k, int(self.labels[i]))


def _collate(batch):
    ids, attn, xs, ks, ys = zip(*batch)
    mods = xs[0].keys()
    feats = {m: torch.stack([x[m] for x in xs]) for m in mods}
    masks = {m: torch.stack([k[m] for k in ks]) for m in mods if m in ks[0]}
    return (torch.stack(ids), torch.stack(attn), feats, masks,
            torch.tensor(ys, dtype=torch.long))


class FusionWithTextEncoder(nn.Module):
    """Wraps a fusion module so text embeddings are produced in the graph.

    The fusion module is exactly the one used in the frozen study
    (`models.LateFusion` / `models.TriCAN`), so switching this study on and off
    changes the protocol and nothing else.
    """

    def __init__(self, kind: str, encoder_name: str, other_dims: dict[str, int],
                 n_labels: int, cfg: FinetuneConfig) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        d_text = self.encoder.config.hidden_size
        dims = {"text": d_text, **other_dims}

        kw: dict = {"d_model": cfg.d_model, "dropout": cfg.dropout}
        if kind == "tri-can":
            kw["n_heads"] = cfg.n_heads
        self.fusion = M.build(kind, dims, n_labels, **kw)

    def forward(self, input_ids, attention_mask, feats, masks):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        all_feats = {"text": h, **feats}
        all_masks = {"text": attention_mask.bool(), **masks}
        return self.fusion(all_feats, all_masks)

    def param_groups(self, cfg: FinetuneConfig):
        return [
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": self.fusion.parameters(), "lr": cfg.lr_head},
        ]


@torch.no_grad()
def _predict(model: nn.Module, dl: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    for ids, attn, feats, masks, _ in dl:
        out.append(model(ids, attn, feats, masks).argmax(dim=-1).numpy())
    return np.concatenate(out)


def cross_validate_finetuned(
    kind: str,
    texts: list[str],
    feats: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    labels: np.ndarray,
    n_labels: int,
    cfg: FinetuneConfig,
) -> dict:
    """Stratified K-fold with the text encoder trainable.

    Folds come from the same StratifiedKFold(seed) as the frozen study, so the
    two protocols are compared on identical splits.
    """
    from transformers import AutoTokenizer

    torch.set_num_threads(cfg.n_threads)
    tok = AutoTokenizer.from_pretrained(cfg.encoder)
    enc = tok([t or "" for t in texts], padding="max_length", truncation=True,
              max_length=cfg.max_len, return_tensors="pt")
    enc = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    other = {m: feats[m] for m in feats if m != "text"}
    other_masks = {m: masks[m] for m in masks if m != "text"}
    other_dims = {m: int(other[m].shape[-1]) for m in other}

    n = len(labels)
    oof = np.full(n, -1, dtype=int)
    folds = []
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    for k, (tr, va) in enumerate(skf.split(np.zeros(n), labels)):
        seed = cfg.seed * 1000 + k
        _seed_everything(seed)

        model = FusionWithTextEncoder(kind, cfg.encoder, other_dims, n_labels, cfg)
        sub = lambda d, idx: {m: d[m][idx] for m in d}
        tr_ds = TokenizedDataset({kk: v[tr] for kk, v in enc.items()},
                                 sub(other, tr), sub(other_masks, tr), labels[tr])
        va_ds = TokenizedDataset({kk: v[va] for kk, v in enc.items()},
                                 sub(other, va), sub(other_masks, va), labels[va])
        tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=_collate)
        va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=_collate)

        crit = _make_loss(type("C", (), {"loss": "weighted_ce", "focal_gamma": 2.0})(),
                          labels[tr], n_labels)
        opt = torch.optim.AdamW(model.param_groups(cfg), weight_decay=cfg.weight_decay)
        total = max(len(tr_dl) * cfg.epochs, 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[cfg.lr_encoder, cfg.lr_head], total_steps=total,
            pct_start=cfg.warmup_frac, anneal_strategy="linear")

        best_f1, best_state = -1.0, None
        for _ in range(cfg.epochs):
            model.train()
            for ids, attn, f, m, y in tr_dl:
                opt.zero_grad()
                loss = crit(model(ids, attn, f, m), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

            yp = _predict(model, va_dl)
            f1 = float(f1_score(labels[va], yp, average="macro", zero_division=0))
            if f1 > best_f1:
                best_f1, best_state = f1, copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
        oof[va] = _predict(model, va_dl)
        folds.append({"fold": k, "best_val_macro_f1": best_f1, "seed": seed})
        print(f"    fold {k}: val macro-F1 {best_f1:.4f}", flush=True)

    if (oof < 0).any():
        raise RuntimeError("some samples received no out-of-fold prediction")
    return {"model": kind, "y_pred": oof, "folds": folds}
