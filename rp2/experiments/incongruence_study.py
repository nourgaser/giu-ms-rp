"""
incongruence_study.py — the one target in Ar-MuSA on which fusion can matter.

Motivation, in one paragraph. The main experiment returned a null: cross-modal
attention gave no advantage over late fusion in predicting Ar-MuSA's overall
sentiment label. The diagnosis is that the target is not really multimodal —
the overall label matches the corpus's OWN TRANSCRIPT annotation 96.5% of the
time (audio 77.1%, frame 48.2%), so a model that perfectly predicted the text
label would score 0.9655 macro-F1. A text-only model already holds almost all
the label information, which leaves no headroom in which one fusion mechanism
could distinguish itself from another. That is a property of the benchmark, and
it applies to every published result on it, not only to ours.

So this asks a different question of the same data:

    Can the model tell whether the modalities AGREE with each other?

This target cannot be read off any single modality even in principle: nothing in
the transcript says what the speaker's voice sounded like. A text-only model
should therefore sit near its class prior, and any model that beats it is
demonstrably using cross-modal information. Cross-modal attention is explicitly
a mechanism for letting one modality inspect another, so if it is worth having,
this is where it shows.

Reported honestly as a SECONDARY analysis. It does not undo the primary null;
it localises it. And the prediction is made before the run, not after:
  * text-only ~ class prior
  * any two modalities > any one
  * tri-can > late-fusion, and > the capacity-matched control

Everything loads once and reuses the same folds, unlike running the CLI per
condition, which reloads a 1.6 GB cache each time and oversubscribes the CPU.

    python -m experiments.incongruence_study --cache data/cache/ar-musa
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import models as M                                   # noqa: E402
from rp2.evaluate import core_metrics                         # noqa: E402
from rp2.features import load_cache                           # noqa: E402
from rp2.train import TrainConfig, cross_validate             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/ar-musa"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/incongruence.json"))
    args = ap.parse_args()

    feats, masks, _, meta, manifest = load_cache(args.cache)
    cong = meta["congruence"]
    y = np.array([0 if c == "congruent" else 1 for c in cong])
    n = len(y)
    prior = max(1 - y.mean(), y.mean())

    print(f"target     : do the modalities disagree?")
    print(f"samples    : {n}   congruent {int((y == 0).sum())} / "
          f"incongruent {int((y == 1).sum())}")
    print(f"majority-class accuracy (the number to beat): {prior:.4f}")
    print(f"encoders   : {manifest['config']['encoders']}\n")

    cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=args.seed,
                      n_threads=args.threads)
    results: dict[str, dict] = {}

    def run(tag: str, kind: str, mods: tuple[str, ...]) -> dict:
        f = {m: feats[m] for m in mods}
        k = {m: masks[m] for m in mods if m in masks}
        t0 = time.time()
        out = cross_validate(kind, f, y, 2, cfg, k)
        m = core_metrics(y, out["y_pred"], 2)
        m["params"] = out["params"]
        m["seconds"] = time.time() - t0
        results[tag] = m
        print(f"  {tag:34s} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"({m['params']:,} params, {m['seconds']:.0f}s)", flush=True)
        return m

    print("unimodal — how much can one modality alone know?")
    for mod in ("text", "audio", "visual"):
        run(f"late-fusion / {mod}", "late-fusion", (mod,))

    print("\nbimodal — does a second modality help?")
    for a, b in itertools.combinations(("text", "audio", "visual"), 2):
        run(f"late-fusion / {a}+{b}", "late-fusion", (a, b))

    print("\ntrimodal — and does the fusion MECHANISM matter?")
    allm = ("text", "audio", "visual")
    for kind in ("late-fusion", "late-fusion-wide", "tri-can"):
        run(f"{kind} / all", kind, allm)

    # -- the comparisons the argument rests on ----------------------------
    uni_best = max(results[f"late-fusion / {m}"]["macro_f1"]
                   for m in ("text", "audio", "visual"))
    lf = results["late-fusion / all"]["macro_f1"]
    lfw = results["late-fusion-wide / all"]["macro_f1"]
    tc = results["tri-can / all"]["macro_f1"]

    print("\n" + "=" * 70)
    print(f"best single modality           : {uni_best:.4f}")
    print(f"late fusion, all three         : {lf:.4f}   ({lf - uni_best:+.4f} vs best unimodal)")
    print(f"capacity-matched late fusion   : {lfw:.4f}")
    print(f"TRI-CAN                        : {tc:.4f}")
    print(f"  tri-can - late-fusion        : {tc - lf:+.4f}")
    print(f"  tri-can - capacity-matched   : {tc - lfw:+.4f}   <- the one that counts")
    print("=" * 70)

    if tc - lfw > 0.01:
        print("\nCross-modal attention beats an equally-sized late-fusion model on a\n"
              "target that requires comparing modalities. That is a positive result\n"
              "for the mechanism, and it is consistent with the primary null: the\n"
              "mechanism works, the usual benchmark target just cannot show it.")
    else:
        print("\nNo advantage over the capacity-matched control even here. On this\n"
              "corpus and these frozen features, cross-modal attention is not\n"
              "earning its parameters. Report it as such.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"target": "incongruence", "n": int(n), "prior_accuracy": float(prior),
         "encoders": manifest["config"]["encoders"], "results": results},
        indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
