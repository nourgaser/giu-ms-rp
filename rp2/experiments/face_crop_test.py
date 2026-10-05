"""
face_crop_test.py — does cropping to the face help the expression encoder?

The visual sweep showed that swapping Swin-tiny (an ImageNet OBJECT classifier)
for a facial-expression-pretrained ViT is worth +0.0263 macro-F1 on EGY-MER,
where vision is the strongest modality. But that comparison fed both encoders
full talk-show frames, and FER models are trained on tight, centred face crops.
Most of the expression model's capacity was being spent on background.

This measures the difference directly: identical encoder, identical samples,
identical folds, with and without cropping to the detected face.

Detection is OpenCV's YuNet. Frames with no detected face keep the full frame,
so a miss is never worse than not cropping, and the hit rate is reported rather
than assumed.

    python -m experiments.face_crop_test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2.data import get_dataset                              # noqa: E402
from rp2.evaluate import core_metrics                         # noqa: E402
from rp2.features import (ExtractionConfig, FeatureExtractor,  # noqa: E402
                          SEQ_LEN, load_cache)
from rp2.train import TrainConfig, cross_validate             # noqa: E402

FER = "trpakov/vit-face-expression"
SWIN_BASELINE = 0.5225      # Swin-tiny, full frames, same protocol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="egy-mer")
    ap.add_argument("--root", type=Path, default=Path("data/raw/egy-mer"))
    ap.add_argument("--encoder", default=FER)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("results/face_crop_test.json"))
    args = ap.parse_args()

    samples = list(get_dataset(args.dataset, args.root).samples())
    print(f"{len(samples)} samples   encoder {args.encoder}   "
          f"{args.frames} frames/segment\n")

    rows = {}
    for crop in (False, True):
        tag = "face-crop" if crop else "full-frame"
        out = Path(f"data/cache/facecrop-{tag}")
        cfg = ExtractionConfig(
            encoders={"text": "", "audio": "", "visual": args.encoder},
            seq_len=dict(SEQ_LEN), face_crop=crop, visual_frames=args.frames)
        fx = FeatureExtractor(cfg, modalities=("visual",))
        if not (out / "manifest.json").exists():
            fx.extract(samples, out, log_every=200)
        hit = getattr(fx, "faces_detected", 0)
        seen = getattr(fx, "frames_seen", 0)

        feats, masks, labels, _, _ = load_cache(out, ("visual",))
        n_labels = int(labels.max()) + 1
        scores = []
        for seed in args.seeds:
            tc = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=seed,
                             n_threads=args.threads)
            r = cross_validate("late-fusion", feats, labels, n_labels, tc, masks)
            scores.append(core_metrics(labels, r["y_pred"], n_labels)["macro_f1"])

        m, s = float(np.mean(scores)), float(np.std(scores))
        rows[tag] = {"macro_f1": m, "std": s, "per_seed": scores,
                     "faces_detected": hit, "frames_seen": seen}
        extra = (f"   faces found {hit}/{seen} = {hit / max(seen, 1):.0%}"
                 if crop else "")
        print(f"  {tag:11s} macro-F1 {m:.4f} +/- {s:.4f}{extra}", flush=True)
        if not args.keep_cache:
            shutil.rmtree(out, ignore_errors=True)

    d = rows["face-crop"]["macro_f1"] - rows["full-frame"]["macro_f1"]
    print(f"\ncropping gain            : {d:+.4f}")
    print(f"vs Swin-tiny full frames : "
          f"{rows['face-crop']['macro_f1'] - SWIN_BASELINE:+.4f}   "
          f"(Swin {SWIN_BASELINE:.4f})")
    spread = max(r["std"] for r in rows.values())
    if abs(d) <= spread:
        print(f"NOTE: the gap is within the seed spread ({spread:.4f}); "
              f"treat it as a tie, not a win.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": args.dataset, "encoder": args.encoder,
         "frames": args.frames, "seeds": args.seeds,
         "swin_baseline": SWIN_BASELINE, "results": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
