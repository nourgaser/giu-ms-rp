"""
encoder_sweep.py — which frozen encoder actually represents Egyptian Arabic?

Why this exists. The first full run produced a null result: Tri-CAN did not beat
late fusion. The unimodal ablations showed why — text alone reached macro-F1
0.5458 and all three modalities together reached 0.5593, so audio and visual
contributed about +0.0135 between them. There was nothing to fuse.

But "the modalities contribute nothing" is only a finding about the TASK if the
encoders are a fair representation of each modality. They were not:

  * audio   `facebook/wav2vec2-base` is pretrained on English (LibriSpeech).
            Using it on Egyptian Arabic and concluding "audio carries little
            signal" confounds the modality with the encoder.
  * text    `aubmindlab/bert-base-arabertv2` is trained predominantly on
            Modern Standard Arabic. Ar-MuSA is Egyptian dialect — different
            orthography, vocabulary and morphology. MARBERT and CAMeLBERT-DA
            are trained on dialectal Arabic and should fit far better.
  * visual  `microsoft/swin-tiny` is an ImageNet OBJECT classifier. Facial
            affect is not an ImageNet category. A face- or expression-trained
            model, or CLIP's semantic space, is the right comparison.

So before any claim about fusion, each modality gets the best frozen encoder we
can cheaply find. Otherwise the study measures our defaults, not the problem.

Method: for one modality, extract features with each candidate encoder over the
same subset, train the same unimodal model over the same folds, and compare
macro-F1. A subset keeps each candidate to a few minutes; the winner is then
re-extracted over the full corpus.

    python -m experiments.encoder_sweep --modality text --n 3000
    python -m experiments.encoder_sweep --modality visual --n 3000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2.data import get_dataset                               # noqa: E402
from rp2.evaluate import core_metrics                          # noqa: E402
from rp2.features import (ExtractionConfig, FeatureExtractor,   # noqa: E402
                          SEQ_LEN, load_cache)
from rp2.train import TrainConfig, cross_validate              # noqa: E402

CANDIDATES = {
    "text": [
        # MSA-centric; the current default and the baseline to beat.
        "aubmindlab/bert-base-arabertv2",
        # 1B Arabic tweets, heavily dialectal — the natural fit for Egyptian.
        "UBC-NLP/MARBERT",
        # CAMeL's dialectal-Arabic variant.
        "CAMeL-Lab/bert-base-arabic-camelbert-da",
        # AraBERT retrained on Twitter; dialect-aware sibling of the default.
        "aubmindlab/bert-base-arabertv02-twitter",
    ],
    "audio": [
        "facebook/wav2vec2-base",              # English — the current default
        # Pretrained FOR speech emotion recognition (SUPERB ER). The task here
        # is emotion; a generic self-supervised speech model is not the
        # obvious choice and was never compared against one that is.
        "superb/wav2vec2-base-superb-er",
    ],
    "visual": [
        "microsoft/swin-tiny-patch4-window7-224",   # ImageNet objects — default
        # Pretrained on facial expression recognition. Facial affect is not an
        # ImageNet category, and vision is the STRONGEST modality on EGY-MER,
        # so this is the largest single untested lever in the study.
        "trpakov/vit-face-expression",
        "dima806/facial_emotions_image_detection",
        "openai/clip-vit-base-patch32",             # semantic, not object-label
    ],
}


def evaluate(modality: str, encoder: str, samples, out: Path, args) -> dict | None:
    """Extract with one encoder, train a unimodal model, return its metrics."""
    if not (out / "manifest.json").exists():
        cfg = ExtractionConfig(
            encoders={m: (encoder if m == modality else "")
                      for m in ("text", "audio", "visual")},
            seq_len=dict(SEQ_LEN),
        )
        FeatureExtractor(cfg, modalities=(modality,)).extract(
            samples, out, log_every=500)

    feats, masks, labels, _, _ = load_cache(out, (modality,))
    n_labels = int(labels.max()) + 1
    cfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=0,
                      n_threads=args.threads)
    res = cross_validate("late-fusion", feats, labels, n_labels, cfg, masks)
    m = core_metrics(labels, res["y_pred"], n_labels)
    m["dim"] = int(feats[modality].shape[-1])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=("text", "audio", "visual"))
    ap.add_argument("--dataset", default="ar-musa")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache/sweep"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep per-encoder feature caches (disk is tight)")
    args = ap.parse_args()

    encoders = args.encoders or CANDIDATES[args.modality]
    root = args.root or Path(f'data/raw/{args.dataset}')
    samples = list(get_dataset(args.dataset, root).samples())[: args.n]
    print(f"sweeping {len(encoders)} {args.modality} encoders "
          f"on {len(samples)} samples\n")

    rows = []
    for enc in encoders:
        out = args.cache_root / f"{args.modality}-{enc.replace('/', '_')}"
        print(f"--- {enc}")
        try:
            m = evaluate(args.modality, enc, samples, out, args)
            rows.append((enc, m))
            print(f"    macro-F1 {m['macro_f1']:.4f}  (dim {m['dim']})\n", flush=True)
        except Exception:
            # A candidate that will not load is a fact about the candidate, not
            # a reason to abandon the sweep.
            print(f"    FAILED:\n{traceback.format_exc(limit=2)}", flush=True)
        finally:
            if not args.keep_cache and out.exists():
                shutil.rmtree(out, ignore_errors=True)

    if not rows:
        print("no encoder succeeded", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: -r[1]["macro_f1"])
    print(f"\n{'encoder':46s} {'macro-F1':>9s} {'wt-F1':>8s} {'acc':>7s} {'dim':>5s}")
    print("-" * 80)
    for enc, m in rows:
        print(f"{enc:46s} {m['macro_f1']:9.4f} {m['weighted_f1']:8.4f} "
              f"{m['accuracy']:7.4f} {m['dim']:5d}")

    best, base = rows[0], next((r for r in rows if r[0] == CANDIDATES[args.modality][0]), None)
    print(f"\nbest: {best[0]}  macro-F1 {best[1]['macro_f1']:.4f}")
    if base and best[0] != base[0]:
        print(f"  vs current default ({base[0]}): "
              f"{best[1]['macro_f1'] - base[1]['macro_f1']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
