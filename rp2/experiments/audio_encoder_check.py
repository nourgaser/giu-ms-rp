"""
audio_encoder_check.py — is the weak audio signal the encoder's fault?

The main result is a null: cross-modal attention gives no advantage over late
fusion on Ar-MuSA. The unimodal ablations explain why — text alone reaches
macro-F1 0.5458 and all three modalities together reach 0.5593, so the
non-verbal streams contribute about +0.0135. There is almost nothing to fuse,
and no fusion mechanism can distinguish itself on that margin.

That invites the obvious objection: the default audio encoder,
`facebook/wav2vec2-base`, is pretrained on English (LibriSpeech). Of course it
is weak on Egyptian Arabic. Maybe a multilingual encoder would give audio real
signal and the whole picture would change.

This script answers that with a measurement instead of an argument. It
re-extracts audio features for a subset using `wav2vec2-xls-r-300m` (128
languages, Arabic included) and compares audio-only macro-F1 against the
English model ON THE SAME SAMPLES AND THE SAME FOLDS.

A subset rather than the full corpus because this is a screening question: if
XLS-R is substantially better on 2,000 samples it justifies a full re-run, and
if it is not, the objection is answered for the price of 30 minutes rather
than 3 hours. Deciding by measurement is the point.

    python -m experiments.audio_encoder_check --n 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2.data import ArMuSA                                   # noqa: E402
from rp2.features import (ExtractionConfig, FeatureExtractor,  # noqa: E402
                          SEQ_LEN, load_cache)
from rp2.train import TrainConfig, cross_validate              # noqa: E402
from rp2.evaluate import core_metrics                          # noqa: E402

XLSR = "facebook/wav2vec2-xls-r-300m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ar-musa"))
    ap.add_argument("--base-cache", type=Path, default=Path("data/cache/ar-musa"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/ar-musa-xlsr-audio"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--encoder", default=XLSR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    samples = list(ArMuSA(args.root).samples())[: args.n]
    print(f"comparing audio encoders on the first {len(samples)} samples\n")

    # -- XLS-R features (extract once, reuse if already cached) -----------
    if not (args.out / "manifest.json").exists():
        cfg = ExtractionConfig(
            encoders={"text": "", "audio": args.encoder, "visual": ""},
            seq_len=dict(SEQ_LEN),
        )
        FeatureExtractor(cfg, modalities=("audio",)).extract(
            samples, args.out, log_every=100)
    else:
        print(f"(reusing cache at {args.out})")

    xf, xm, xlabels, _, xman = load_cache(args.out, ("audio",))

    # -- the same samples from the English-encoder cache ------------------
    bf, bm, blabels, _, bman = load_cache(args.base_cache, ("audio",))
    bf = {"audio": bf["audio"][: len(samples)]}
    bm = {"audio": bm["audio"][: len(samples)]}
    blabels = blabels[: len(samples)]

    if not np.array_equal(blabels, xlabels):
        print("ERROR: label vectors differ between caches; the two runs are not "
              "on the same samples, so the comparison would be meaningless.",
              file=sys.stderr)
        return 2

    n_labels = int(max(xlabels.max(), blabels.max())) + 1
    cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=0,
                      n_threads=args.threads)

    rows = []
    for name, feats, masks, enc in (
        ("wav2vec2-base (English)", bf, bm, bman["config"]["encoders"]["audio"]),
        ("XLS-R-300m (multilingual)", xf, xm, args.encoder),
    ):
        out = cross_validate("late-fusion", feats, xlabels, n_labels, cfg, masks)
        m = core_metrics(xlabels, out["y_pred"], n_labels)
        rows.append((name, enc, m["macro_f1"], m["weighted_f1"], m["accuracy"]))

    print(f"\n{'audio encoder':30s} {'macro-F1':>9s} {'wt-F1':>8s} {'acc':>7s}")
    print("-" * 58)
    for name, enc, f1, wf1, acc in rows:
        print(f"{name:30s} {f1:9.4f} {wf1:8.4f} {acc:7.4f}")
    delta = rows[1][2] - rows[0][2]
    print(f"\nXLS-R minus base: {delta:+.4f} macro-F1 (audio only, n={len(samples)})")
    print("chance for 3 classes is ~0.333")
    if delta > 0.05:
        print("\n-> Substantial. A full XLS-R re-extraction is justified before "
              "concluding anything about fusion.")
    else:
        print("\n-> Not substantial. The weak audio contribution is a property of "
              "the task/corpus as encoded, not an artefact of using an "
              "English-pretrained encoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
