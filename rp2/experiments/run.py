"""
run.py — the experiment.

    # no data needed: proves the pipeline end to end in ~1 minute
    python -m experiments.run --dataset synthetic

    # the real study, once features are cached
    python -m experiments.run --dataset ar-musa --cache data/cache/ar-musa

    # when/if EGY-MER arrives, the same command with one word changed
    python -m experiments.run --dataset egy-mer --cache data/cache/egy-mer

Runs all three models over identical folds, then reports:
  * overall macro/weighted F1 per model,
  * the same broken down by congruence stratum,
  * the difference-in-differences with a bootstrap CI — the headline,
  * the sarcasm slice where the corpus provides the flag.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import data as rp2data                      # noqa: E402
from rp2 import evaluate as ev                       # noqa: E402
from rp2 import models as M                          # noqa: E402
from rp2.features import load_cache                  # noqa: E402
from rp2.train import TrainConfig, cross_validate    # noqa: E402


def load_synthetic(cfg_signal: float, n: int, seed: int):
    """Synthetic features, generated in memory. No encoders, no disk."""
    corpus = rp2data.SyntheticCorpus(n=n, signal=cfg_signal, seed=seed)
    samples = list(corpus.samples())
    dims = {"text": 64, "audio": 64, "visual": 64}
    feats = corpus.features(dims)
    masks = {m: np.ones(feats[m].shape[:2], dtype=bool) for m in feats}
    labels = np.array([s.label for s in samples])
    meta = {
        "uid": [s.uid for s in samples],
        "congruence": [s.congruence for s in samples],
        "sarcasm": [s.meta.get("sarcasm") for s in samples],
    }
    return feats, masks, labels, meta, list(corpus.labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--cache", type=Path, default=None,
                    help="feature cache directory (required for real corpora)")
    ap.add_argument("--target", default="overall",
                    choices=("overall", "incongruence", "sarcasm"),
                    help=("what to predict. 'overall' is the corpus label. "
                          "'incongruence' is whether the per-modality "
                          "annotations disagree -- a target that CANNOT be "
                          "read off any single modality, and so is the one "
                          "target in this corpus on which a fusion mechanism "
                          "has room to distinguish itself. 'sarcasm' is the "
                          "corpus's own overall sarcasm flag."))
    ap.add_argument("--models", nargs="+", default=list(M.MODELS))
    ap.add_argument("--modalities", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "focal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    # synthetic-only knobs
    ap.add_argument("--synthetic-n", type=int, default=900)
    # 0.03 rather than something larger: measured sweep (see PLAN.md §7) shows
    # the synthetic task saturates at macro-F1 1.0 above ~0.1, which makes it
    # useless for telling models apart. 0.0 is the null control.
    ap.add_argument("--synthetic-signal", type=float, default=0.03)
    ap.add_argument("--list", action="store_true", help="list known corpora and exit")
    args = ap.parse_args()

    if args.list:
        print("known corpora:\n" + rp2data.describe())
        return 0

    # -- load ------------------------------------------------------------
    if args.dataset == "synthetic":
        feats, masks, labels, meta, label_names = load_synthetic(
            args.synthetic_signal, args.synthetic_n, args.seed)
    else:
        if args.cache is None or not args.cache.exists():
            print(f"error: --cache is required for {args.dataset} and must exist.\n"
                  f"Run:  python -m experiments.extract --dataset {args.dataset} ...",
                  file=sys.stderr)
            return 2
        feats, masks, labels, meta, manifest = load_cache(args.cache, args.modalities)
        adapter = rp2data.REGISTRY[args.dataset][0]
        label_names = list(getattr(adapter, "labels", []) or
                           [str(i) for i in range(int(labels.max()) + 1)])

    if args.modalities:
        feats = {m: feats[m] for m in args.modalities}
        masks = {m: masks[m] for m in args.modalities if m in masks}

    # -- retarget ----------------------------------------------------------
    # The default target on Ar-MuSA is 96.5% recoverable from the transcript
    # annotation alone, so it cannot discriminate fusion mechanisms: a text-only
    # model already has almost all the label information. These alternatives
    # are targets a single modality demonstrably cannot supply.
    if args.target == "incongruence":
        cong_raw = meta.get("congruence") or []
        if not cong_raw or all(c is None for c in cong_raw):
            print("error: --target incongruence needs per-modality labels, "
                  "which this corpus does not provide.", file=sys.stderr)
            return 2
        labels = np.array([0 if c == "congruent" else 1 for c in cong_raw])
        label_names = ["congruent", "incongruent"]
    elif args.target == "sarcasm":
        sar = meta.get("sarcasm") or []
        if not sar or all(s is None for s in sar):
            print("error: --target sarcasm needs the corpus sarcasm flag.",
                  file=sys.stderr)
            return 2
        keep = np.array([s is not None for s in sar])
        labels = np.array([int(s) for s in sar if s is not None])
        feats = {m: feats[m][keep] for m in feats}
        masks = {m: masks[m][keep] for m in masks}
        meta = {k: [v for v, k2 in zip(vals, keep) if k2]
                for k, vals in meta.items() if isinstance(vals, list)}
        label_names = ["not_sarcastic", "sarcastic"]

    n_labels = len(label_names) or int(labels.max()) + 1
    print(f"corpus     : {args.dataset}  target: {args.target}")
    print(f"samples    : {len(labels)}")
    print(f"labels     : {n_labels} {label_names}")
    print(f"modalities : {sorted(feats)}  dims "
          f"{ {m: feats[m].shape[-1] for m in sorted(feats)} }")
    dist = np.bincount(labels, minlength=n_labels)
    print(f"class dist : {dict(zip(label_names, dist.tolist()))}")

    cong = meta.get("congruence") or [None] * len(labels)
    if args.target == "incongruence":
        # The stratum and the label would be the same variable; a
        # congruence-stratified report on this target is circular by
        # construction, so it is switched off rather than printed and
        # misread later.
        cong = [None] * len(labels)
    n_cong = sum(1 for c in cong if c == "congruent")
    n_inc = sum(1 for c in cong if c not in (None, "congruent"))
    if n_cong and n_inc:
        print(f"congruence : {n_cong} congruent / {n_inc} incongruent "
              f"({n_inc/(n_cong+n_inc):.1%} incongruent)")
    else:
        print("congruence : unavailable for this corpus "
              "(no per-modality labels) — DiD analysis will be skipped")

    cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, d_model=args.d_model,
                      dropout=args.dropout, lr=args.lr, loss=args.loss,
                      seed=args.seed, n_threads=args.threads)

    # -- run ---------------------------------------------------------------
    preds: dict[str, ev.Predictions] = {}
    results: dict[str, dict] = {}
    param_counts: dict[str, int] = {}

    for kind in args.models:
        t0 = time.time()
        print(f"\n=== {kind} " + "=" * (50 - len(kind)))
        out = cross_validate(kind, feats, labels, n_labels, cfg, masks)
        param_counts[kind] = out["params"]
        p = ev.Predictions(model=kind, y_true=labels, y_pred=out["y_pred"],
                           uids=list(meta["uid"]), congruence=list(cong),
                           sarcasm=list(meta.get("sarcasm") or []),
                           fold=out["fold"])
        preds[kind] = p
        results[kind] = ev.by_stratum(p, n_labels)
        print(f"  params {out['params']:,}   "
              f"overall macro-F1 {results[kind]['overall']['macro_f1']:.4f}   "
              f"({time.time()-t0:.1f}s)")

    # -- report ------------------------------------------------------------
    print("\n" + "=" * 62)
    print("PARAMETER COUNTS (capacity control)")
    for k, v in param_counts.items():
        print(f"  {k:20s} {v:>12,}")

    did = None
    if "late-fusion" in preds and "tri-can" in preds and n_cong and n_inc:
        did = ev.difference_in_differences(preds["late-fusion"], preds["tri-can"],
                                           n_boot=args.boot, seed=args.seed)
    print(ev.format_report(results, did))

    if "late-fusion-wide" in preds and "tri-can" in preds and n_cong and n_inc:
        did_w = ev.difference_in_differences(preds["late-fusion-wide"],
                                             preds["tri-can"],
                                             n_boot=args.boot, seed=args.seed)
        print("against the CAPACITY-MATCHED baseline (late-fusion-wide):")
        print(f"  DiD {did_w['did']:+.4f}  CI95 "
              f"[{did_w['ci95'][0]:+.4f}, {did_w['ci95'][1]:+.4f}]\n")
    else:
        did_w = None

    sarc = {k: ev.sarcasm_slice(p, n_labels) for k, p in preds.items()}
    if any(sarc.values()):
        print("sarcasm slice (corpus flag)")
        print("-" * 62)
        for k, s in sarc.items():
            for name, m in s.items():
                print(f"  {k + ' / ' + name:34s} n={m['n']:5d}  "
                      f"macro-F1 {m['macro_f1']:.4f}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = (f"{args.dataset}-{args.target}-seed{args.seed}"
             if args.target != "overall" else f"{args.dataset}-seed{args.seed}")
    payload = {
        "dataset": args.dataset, "n": int(len(labels)),
        "label_names": label_names, "config": vars(cfg),
        "params": param_counts, "results": results,
        "did_vs_late_fusion": did, "did_vs_capacity_matched": did_w,
        "sarcasm": sarc,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    path = args.out / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    np.savez_compressed(args.out / f"{stamp}-predictions.npz",
                        y_true=labels,
                        **{f"pred_{k}": p.y_pred for k, p in preds.items()},
                        congruence=np.array([str(c) for c in cong]))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
