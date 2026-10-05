"""
frozen_vs_finetuned.py — does the fusion conclusion survive better features?

The whole study so far uses frozen encoders. That protocol is defensible (it
isolates the fusion mechanism perfectly) but it caps text-only macro-F1 near
0.63 while published Ar-MuSA work reaches 0.78-0.90 by fine-tuning. If
cross-modal attention only pays once representations are strong, a frozen study
could never detect it and the null would be an artefact of the protocol.

This runs both arms on identical folds, with identical fusion modules, and the
SAME text encoder in both, so the only thing that changes is whether that
encoder's weights move:

    frozen       text/audio/visual features precomputed and fixed
    fine-tuned   text encoder trainable; audio and visual still frozen

Audio and visual stay frozen in both arms. That is a CPU-budget limitation, not
a claim -- backprop through Wav2Vec2 over 8 s of audio is far beyond this
machine. It is reported as a limitation.

    python -m experiments.frozen_vs_finetuned --target overall
    python -m experiments.frozen_vs_finetuned --target incongruence
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import models as M                                    # noqa: E402
from rp2.data import ArMuSA                                    # noqa: E402
from rp2.evaluate import core_metrics                          # noqa: E402
from rp2.features import (ExtractionConfig, FeatureExtractor,   # noqa: E402
                          SEQ_LEN, load_cache)
from rp2.finetune import FinetuneConfig, cross_validate_finetuned   # noqa: E402
from rp2.train import TrainConfig, cross_validate               # noqa: E402

DEFAULT_TEXT_ENCODER = "aubmindlab/bert-base-arabertv02-twitter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ar-musa"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/ar-musa"))
    ap.add_argument("--text-cache", type=Path,
                    default=Path("data/cache/ar-musa-text-twitter"))
    ap.add_argument("--encoder", default=DEFAULT_TEXT_ENCODER)
    ap.add_argument("--target", default="overall",
                    choices=("overall", "incongruence"))
    ap.add_argument("--models", nargs="+", default=["late-fusion", "tri-can"])
    ap.add_argument("--modalities", nargs="+", default=None,
                    help=("restrict to these modalities. `--modalities text` "
                          "gives the fine-tuned TEXT-ONLY baseline, which is "
                          "the control that matters: if it matches fine-tuned "
                          "late fusion, then the non-verbal streams add nothing "
                          "even once the representations are strong, and the "
                          "null is about the corpus rather than the protocol."))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--frozen-epochs", type=int, default=40)
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N samples (for a quick check)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # -- text features from the chosen encoder ----------------------------
    samples = list(ArMuSA(args.root).samples())
    if not (args.text_cache / "manifest.json").exists():
        print(f"extracting text features with {args.encoder} ...")
        cfg = ExtractionConfig(
            encoders={"text": args.encoder, "audio": "", "visual": ""},
            seq_len=dict(SEQ_LEN))
        FeatureExtractor(cfg, modalities=("text",)).extract(
            samples, args.text_cache, log_every=1000)

    tf, tm, tlabels, tmeta, _ = load_cache(args.text_cache, ("text",))
    af, am, alabels, ameta, _ = load_cache(args.cache, ("audio", "visual"))

    if list(tmeta["uid"]) != list(ameta["uid"]):
        print("ERROR: the text cache and the audio/visual cache are not aligned "
              "(differing uid order). Refusing to combine them.", file=sys.stderr)
        return 2

    feats = {"text": tf["text"], **af}
    masks = {"text": tm["text"], **am}
    meta = ameta

    if args.target == "incongruence":
        labels = np.array([0 if c == "congruent" else 1
                           for c in meta["congruence"]])
        label_names = ["congruent", "incongruent"]
    else:
        labels = alabels
        label_names = ["negative", "neutral", "positive"]

    if args.modalities:
        missing = [m for m in args.modalities if m not in feats]
        if missing:
            print(f"error: no cached features for {missing}", file=sys.stderr)
            return 2
        feats = {m: feats[m] for m in args.modalities}
        masks = {m: masks[m] for m in args.modalities if m in masks}
        if "text" not in feats:
            print("error: the fine-tuned arm trains the TEXT encoder, so 'text' "
                  "must be among --modalities.", file=sys.stderr)
            return 2
        if len(feats) < 2 and "tri-can" in args.models:
            # Cross-modal attention needs something to attend TO. Drop it
            # rather than construct a degenerate model that would silently be
            # a plain classifier under a misleading name.
            print("note: tri-can needs >= 2 modalities; running late-fusion only.")
            args.models = [m for m in args.models if m != "tri-can"] or ["late-fusion"]

    by_uid = {s.uid: (s.text or "") for s in samples}
    texts = [by_uid.get(u, "") for u in meta["uid"]]

    if args.limit:
        sl = slice(0, args.limit)
        feats = {m: feats[m][sl] for m in feats}
        masks = {m: masks[m][sl] for m in masks}
        labels, texts = labels[sl], texts[sl]

    n_labels = len(label_names)
    print(f"\ntarget     : {args.target}   ({n_labels} classes)")
    print(f"samples    : {len(labels)}")
    print(f"encoder    : {args.encoder}")
    print(f"folds      : {args.folds}  (identical across both arms)\n")

    results: dict[str, dict] = {}

    # -- arm 1: frozen ----------------------------------------------------
    print("=" * 68)
    print("FROZEN — encoders fixed, only the fusion block trains")
    print("=" * 68)
    fcfg = TrainConfig(epochs=args.frozen_epochs, n_folds=args.folds,
                       seed=args.seed, n_threads=args.threads)
    for kind in args.models + (["late-fusion-wide"] if "tri-can" in args.models else []):
        t0 = time.time()
        out = cross_validate(kind, feats, labels, n_labels, fcfg, masks)
        m = core_metrics(labels, out["y_pred"], n_labels)
        m["params"], m["seconds"] = out["params"], time.time() - t0
        results[f"frozen/{kind}"] = m
        print(f"  {kind:20s} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"({m['params']:,} params, {m['seconds']:.0f}s)", flush=True)

    # -- arm 2: fine-tuned ------------------------------------------------
    print("\n" + "=" * 68)
    print("FINE-TUNED — text encoder trainable; audio/visual still frozen")
    print("=" * 68)
    ftcfg = FinetuneConfig(encoder=args.encoder, epochs=args.ft_epochs,
                           n_folds=args.folds, seed=args.seed,
                           n_threads=args.threads)
    for kind in args.models:
        print(f"  {kind}")
        t0 = time.time()
        out = cross_validate_finetuned(kind, texts, feats, masks, labels,
                                       n_labels, ftcfg)
        m = core_metrics(labels, out["y_pred"], n_labels)
        m["seconds"] = time.time() - t0
        results[f"finetuned/{kind}"] = m
        print(f"  {kind:20s} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"({m['seconds']/60:.1f} min)\n", flush=True)

    # -- comparison -------------------------------------------------------
    print("=" * 68)
    print(f"{'condition':34s} {'macro-F1':>9s} {'acc':>8s}")
    print("-" * 68)
    for k, m in results.items():
        print(f"{k:34s} {m['macro_f1']:9.4f} {m['accuracy']:8.4f}")

    def get(k):
        return results.get(k, {}).get("macro_f1")

    print()
    for arm in ("frozen", "finetuned"):
        lf, tc = get(f"{arm}/late-fusion"), get(f"{arm}/tri-can")
        if lf is not None and tc is not None:
            print(f"{arm:10s}  tri-can - late-fusion : {tc - lf:+.4f}")
    lfw = get("frozen/late-fusion-wide")
    if lfw is not None and get("frozen/tri-can") is not None:
        print(f"frozen      tri-can - capacity-matched : "
              f"{get('frozen/tri-can') - lfw:+.4f}")
    for kind in args.models:
        a, b = get(f"frozen/{kind}"), get(f"finetuned/{kind}")
        if a is not None and b is not None:
            print(f"{kind:20s} fine-tuning gain : {b - a:+.4f}")

    out_path = args.out or Path(f"results/frozen_vs_finetuned-{args.target}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"target": args.target, "encoder": args.encoder, "n": int(len(labels)),
         "folds": args.folds, "results": results}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
