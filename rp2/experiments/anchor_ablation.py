"""
anchor_ablation.py — which modality should be the query?

Tri-CAN anchors on TEXT: text supplies the query, audio and vision supply keys
and values. That choice is inherited from the English multimodal-sentiment
literature (MulT and its descendants) and from Arabic sentiment work, where
text is by far the strongest channel and anchoring on it is obviously right.

On EGY-MER it is obviously wrong, and the numbers say so before any model is
trained. Unimodal macro-F1 on this corpus:

    visual  0.5225      text  0.3404      audio  0.4770

Text is the WEAKEST modality, not the strongest. And the corpus's own
annotations agree: the overall label matches the transcript label only 60.4% of
the time, against 91.2% for audio and 83.0% for video.

Anchoring cross-modal attention on the weakest channel means every query vector
is drawn from the least informative stream. That predicts exactly what the main
run found: Tri-CAN helped on congruent samples (+0.0361, where text is at least
consistent with the others) and hurt on incongruent ones (-0.0462, where text is
precisely the channel that is wrong). The difference-in-differences came out
NEGATIVE, -0.0824 — the opposite of the architectural story.

So the hypothesis under test:

    The anchor should be the DOMINANT modality for the task, not text by
    convention. On sentiment that happens to be text; on emotion it is not.

This is a falsifiable, pre-registered prediction: anchoring on visual or audio
should beat anchoring on text, and should do so most on incongruent samples.
If it does not, the anchor is not the explanation and that is reported.

    python -m experiments.anchor_ablation --cache data/cache/egy-mer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import evaluate as ev                                # noqa: E402
from rp2 import models as M                                   # noqa: E402
from rp2.features import load_cache                           # noqa: E402
from rp2.train import TrainConfig, cross_validate             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/egy-mer"))
    ap.add_argument("--dataset", default="egy-mer")
    ap.add_argument("--anchors", nargs="+", default=["text", "audio", "visual"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/anchor_ablation.json"))
    args = ap.parse_args()

    feats, masks, labels, meta, manifest = load_cache(args.cache)
    n_labels = int(labels.max()) + 1
    cong = meta.get("congruence") or [None] * len(labels)
    uids = list(meta["uid"])
    dims = {m: int(feats[m].shape[-1]) for m in feats}

    print(f"corpus     : {args.dataset}   n={len(labels)}   {n_labels} classes")
    n_c = sum(1 for c in cong if c == "congruent")
    n_i = sum(1 for c in cong if c not in (None, "congruent"))
    print(f"congruence : {n_c} congruent / {n_i} incongruent")
    print(f"seeds      : {args.seeds}  (repeated CV — 484 samples is small, so a "
          f"single split is not trustworthy)\n")

    results: dict[str, dict] = {}
    per_seed: dict[str, list[float]] = {}
    preds_by_cfg: dict[str, list[ev.Predictions]] = {}

    def run(tag: str, kind: str, anchor: str | None) -> None:
        scores, preds = [], []
        for seed in args.seeds:
            cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=seed,
                              n_threads=args.threads)
            # cross_validate builds the model internally; anchor is passed
            # through the same kwargs path the CLI uses.
            out = _cv_with_anchor(kind, feats, labels, n_labels, cfg, masks, anchor)
            p = ev.Predictions(tag, labels, out["y_pred"], uids, list(cong))
            preds.append(p)
            scores.append(ev.core_metrics(labels, out["y_pred"], n_labels)["macro_f1"])
        per_seed[tag] = scores
        preds_by_cfg[tag] = preds
        m = float(np.mean(scores)); s = float(np.std(scores))
        strat = ev.by_stratum(preds[0], n_labels)
        results[tag] = {"macro_f1_mean": m, "macro_f1_std": s,
                        "per_seed": scores, "strata_seed0": strat}
        cg = strat.get("congruent", {}).get("macro_f1", float("nan"))
        ic = strat.get("incongruent", {}).get("macro_f1", float("nan"))
        print(f"  {tag:28s} {m:.4f} +/- {s:.4f}   "
              f"(congruent {cg:.4f} / incongruent {ic:.4f})", flush=True)

    print("baseline")
    run("late-fusion", "late-fusion", None)
    print("\ntri-can, by anchor modality")
    for a in args.anchors:
        run(f"tri-can/anchor={a}", "tri-can", a)

    # -- comparisons -------------------------------------------------------
    print("\n" + "=" * 70)
    base = np.mean(per_seed["late-fusion"])
    best_tag, best = max(((t, np.mean(v)) for t, v in per_seed.items()
                          if t.startswith("tri-can")), key=lambda kv: kv[1])
    print(f"late fusion                 : {base:.4f}")
    for a in args.anchors:
        t = f"tri-can/anchor={a}"
        print(f"tri-can, anchor={a:<7s}     : {np.mean(per_seed[t]):.4f}   "
              f"({np.mean(per_seed[t]) - base:+.4f} vs late fusion)")
    print(f"\nbest anchor: {best_tag.split('=')[1]}  ({best:.4f})")
    txt = np.mean(per_seed["tri-can/anchor=text"])
    print(f"  best minus text-anchored  : {best - txt:+.4f}")

    # DiD for the best anchor against late fusion, seed 0 (paired)
    did = None
    if n_c and n_i:
        try:
            did = ev.difference_in_differences(
                preds_by_cfg["late-fusion"][0], preds_by_cfg[best_tag][0],
                n_boot=args.boot, seed=0)
            lo, hi = did["ci95"]
            print(f"\nDiD for {best_tag} vs late fusion (seed 0):")
            print(f"  congruent gap   {did['gap_congruent']:+.4f}")
            print(f"  incongruent gap {did['gap_incongruent']:+.4f}")
            print(f"  DiD             {did['did']:+.4f}   CI95 [{lo:+.4f}, {hi:+.4f}]")
        except ValueError as e:
            print(f"\nDiD unavailable: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": args.dataset, "n": int(len(labels)), "seeds": args.seeds,
         "encoders": manifest["config"]["encoders"],
         "results": results, "did_best_vs_late_fusion": did},
        indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


def _cv_with_anchor(kind, feats, labels, n_labels, cfg, masks, anchor):
    """cross_validate, but able to pass `anchor` to TriCAN.

    Kept as a thin wrapper rather than changing cross_validate's signature,
    because every other experiment in this project calls that function and its
    behaviour must not shift underneath them.
    """
    if kind != "tri-can" or anchor is None:
        return cross_validate(kind, feats, labels, n_labels, cfg, masks)

    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    from rp2.train import FeatureDataset, _collate, _seed_everything, predict, train_one_fold

    if cfg.n_threads:
        torch.set_num_threads(cfg.n_threads)
    dims = {m: int(np.asarray(feats[m]).shape[-1]) for m in feats}
    n = len(labels)
    oof = np.full(n, -1, dtype=int)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    params = 0
    for k, (tr, va) in enumerate(skf.split(np.zeros(n), labels)):
        seed = cfg.seed * 1000 + k
        _seed_everything(seed)
        model = M.TriCAN(dims, n_labels, d_model=cfg.d_model, n_heads=cfg.n_heads,
                         dropout=cfg.dropout, anchor=anchor)
        params = M.count_params(model)
        sub = lambda d, idx: {m: d[m][idx] for m in d} if d else None
        tr_ds = FeatureDataset(sub(feats, tr), labels[tr], sub(masks, tr))
        va_ds = FeatureDataset(sub(feats, va), labels[va], sub(masks, va))
        model, _ = train_one_fold(model, tr_ds, va_ds, cfg, n_labels, seed)
        _, yp = predict(model, DataLoader(va_ds, batch_size=256, shuffle=False,
                                          collate_fn=_collate))
        oof[va] = yp
    return {"model": f"tri-can/{anchor}", "y_pred": oof, "params": params}


if __name__ == "__main__":
    raise SystemExit(main())
