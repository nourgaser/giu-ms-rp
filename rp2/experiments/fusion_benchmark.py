"""
fusion_benchmark.py — every sensible fusion arrangement, one table.

The project reached a point where isolated comparisons were no longer enough.
Tri-CAN as specified (text as query) underperformed plain late fusion on
EGY-MER, and the diagnosis — that it anchors on the corpus's weakest modality —
implies a family of alternatives rather than a single fix. This runs them all
under one protocol so the answer is a table, not an anecdote.

Arrangements, ordered by how much structure they impose:

  late-fusion        pool each modality, concatenate. No interaction at all.
  late-fusion-wide   the same, widened to match Tri-CAN's parameter count.
                     Separates "attention helped" from "more parameters helped".
  gated-fusion       per-modality learned gates. THIS is the strategy the
                     published EGY-MER baseline uses (weighted F1 0.519), so
                     it is the honest thing to compare against; plain
                     concatenation is not that baseline.
  tri-can:text       cross-attention, text queries audio+vision. The proposal
                     as originally specified, inherited from English sentiment.
  tri-can:audio      same mechanism, audio as query.
  tri-can:visual     same mechanism, vision as query.
  all-pairs          every modality attends to every other (the MulT
                     arrangement). Removes the anchor choice entirely.
  early-fusion       one sequence, full self-attention, learned modality
                     embeddings. No anchor, no direction — the control for
                     whether directed structure buys anything.

Every arrangement sees identical cached features, identical folds, and the
identical training loop. Each is run over several seeds because 484 samples
across 7 classes is small enough that one split is not evidence; the table
reports mean and standard deviation, and differences smaller than the seed
spread are described as ties.

    python -m experiments.fusion_benchmark --cache data/cache/egy-mer
    python -m experiments.fusion_benchmark --cache data/cache/ar-musa --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import evaluate as ev                               # noqa: E402
from rp2 import models as M                                  # noqa: E402
from rp2.features import load_cache                          # noqa: E402
from rp2.train import TrainConfig, cross_validate            # noqa: E402

DEFAULT_MODELS = list(M.MODELS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--dataset", default=None, help="label for the report")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    name = args.dataset or args.cache.name
    feats, masks, labels, meta, manifest = load_cache(args.cache)
    n_labels = int(labels.max()) + 1
    cong = list(meta.get("congruence") or [None] * len(labels))
    uids = list(meta["uid"])

    n_c = sum(1 for c in cong if c == "congruent")
    n_i = sum(1 for c in cong if c not in (None, "congruent"))
    print(f"corpus     : {name}   n={len(labels)}   {n_labels} classes")
    print(f"congruence : {n_c} congruent / {n_i} incongruent")
    print(f"seeds      : {args.seeds}   folds: {args.folds}")
    print(f"encoders   : {manifest['config']['encoders']}\n")

    rows: dict[str, dict] = {}
    preds0: dict[str, ev.Predictions] = {}

    for kind in args.models:
        t0 = time.time()
        scores, wscores, cg, ic = [], [], [], []
        params = 0
        for seed in args.seeds:
            cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=seed,
                              n_threads=args.threads)
            out = cross_validate(kind, feats, labels, n_labels, cfg, masks)
            params = out["params"]
            p = ev.Predictions(kind, labels, out["y_pred"], uids, cong)
            if seed == args.seeds[0]:
                preds0[kind] = p
            st = ev.by_stratum(p, n_labels)
            scores.append(st["overall"]["macro_f1"])
            wscores.append(st["overall"]["weighted_f1"])
            if "congruent" in st:
                cg.append(st["congruent"]["macro_f1"])
            if "incongruent" in st:
                ic.append(st["incongruent"]["macro_f1"])

        rows[kind] = {
            "macro_f1": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
            "weighted_f1": float(np.mean(wscores)),
            "congruent": float(np.mean(cg)) if cg else float("nan"),
            "incongruent": float(np.mean(ic)) if ic else float("nan"),
            "params": int(params), "per_seed": scores,
            "seconds": time.time() - t0,
        }
        r = rows[kind]
        print(f"  {kind:20s} macro {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f}  "
              f"wt {r['weighted_f1']:.4f}  cong {r['congruent']:.4f}  "
              f"incong {r['incongruent']:.4f}  ({r['params']:>9,}p, "
              f"{r['seconds']:.0f}s)", flush=True)

    # -- table -------------------------------------------------------------
    order = sorted(rows, key=lambda k: -rows[k]["macro_f1"])
    print("\n" + "=" * 92)
    print(f"{name}: fusion arrangements, mean over {len(args.seeds)} seeds")
    print("=" * 92)
    print(f"{'arrangement':20s} {'macro-F1':>16s} {'wt-F1':>8s} "
          f"{'congruent':>10s} {'incongr.':>10s} {'params':>11s}")
    print("-" * 92)
    for k in order:
        r = rows[k]
        print(f"{k:20s} {r['macro_f1']:9.4f}±{r['macro_f1_std']:.4f} "
              f"{r['weighted_f1']:8.4f} {r['congruent']:10.4f} "
              f"{r['incongruent']:10.4f} {r['params']:11,}")

    base = rows.get("late-fusion", {}).get("macro_f1")
    if base is not None:
        spread = max(r["macro_f1_std"] for r in rows.values())
        print(f"\nvs plain late fusion ({base:.4f}); "
              f"largest seed SD is {spread:.4f}, so treat smaller gaps as ties:")
        for k in order:
            if k == "late-fusion":
                continue
            d = rows[k]["macro_f1"] - base
            tag = "tie" if abs(d) <= spread else ("better" if d > 0 else "worse")
            print(f"  {k:20s} {d:+.4f}   {tag}")

    # -- DiD for the best attention variant --------------------------------
    did = None
    att = [k for k in order if k.startswith("tri-can") or k in
           {"all-pairs", "early-fusion"}]
    if att and "late-fusion" in preds0 and n_c and n_i:
        best = att[0]
        try:
            did = ev.difference_in_differences(preds0["late-fusion"], preds0[best],
                                               n_boot=args.boot, seed=0)
            lo, hi = did["ci95"]
            print(f"\ndifference-in-differences, {best} vs late fusion (seed "
                  f"{args.seeds[0]}):")
            print(f"  congruent gap   {did['gap_congruent']:+.4f}")
            print(f"  incongruent gap {did['gap_incongruent']:+.4f}")
            print(f"  DiD             {did['did']:+.4f}  CI95 "
                  f"[{lo:+.4f}, {hi:+.4f}]")
            did["model"] = best
        except ValueError as e:
            print(f"\nDiD unavailable: {e}")

    out = args.out or Path(f"results/fusion_benchmark-{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": name, "n": int(len(labels)), "n_labels": n_labels,
         "seeds": args.seeds, "folds": args.folds,
         "encoders": manifest["config"]["encoders"],
         "results": rows, "did": did}, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
