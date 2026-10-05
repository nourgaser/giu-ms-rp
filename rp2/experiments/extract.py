"""
extract.py — run the frozen encoders once over a corpus and cache the features.

    # time 50 clips and project the full run before committing to it
    python -m experiments.extract --dataset ar-musa --root data/raw/ar-musa --pilot 50

    # the real thing
    python -m experiments.extract --dataset ar-musa --root data/raw/ar-musa \
        --out data/cache/ar-musa

The pilot exists because the alternative is guessing. Ar-MuSA is 8,700 clips and
this machine is CPU-only; whether that is a 40-minute job or a 12-hour one
decides whether the corpus is used whole or subsampled, and that decision should
rest on a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import data as rp2data                                  # noqa: E402
from rp2.data.base import DataNotAvailable                       # noqa: E402
from rp2.features import (DEFAULT_ENCODERS, ExtractionConfig,    # noqa: E402
                          FeatureExtractor, SEQ_LEN)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--modalities", nargs="+", default=["text", "audio", "visual"])
    ap.add_argument("--pilot", type=int, default=0,
                    help="encode only N samples and report projected full-run time")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--text-encoder", default=DEFAULT_ENCODERS["text"])
    ap.add_argument("--audio-encoder", default=DEFAULT_ENCODERS["audio"])
    ap.add_argument("--visual-encoder", default=DEFAULT_ENCODERS["visual"])
    ap.add_argument("--audio-seconds", type=float, default=8.0)
    # Which transformer layer to read. -1 = final, which is the wrong choice
    # for audio: mid-stack layers carry the paralinguistic content, worth
    # +0.045 on Ar-MuSA and +0.053 on EGY-MER.
    ap.add_argument("--audio-layer", type=int, default=-1)
    ap.add_argument("--text-layer", type=int, default=-1)
    ap.add_argument("--visual-frames", type=int, default=16,
                    help="frames sampled per video segment (video corpora only)")
    ap.add_argument("--face-crop", action="store_true",
                    help="crop to the detected face before encoding; measured "
                         "as HARMFUL on EGY-MER (see results/face_crop_test.json) "
                         "because the largest face is not always the speaker")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    ds = rp2data.get_dataset(args.dataset, args.root)
    if not ds.is_available():
        note = rp2data.REGISTRY[args.dataset][1]
        print(f"error: {args.dataset} is not available at {args.root}\n  {note}",
              file=sys.stderr)
        return 2

    try:
        samples = list(ds.samples())
    except DataNotAvailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"{args.dataset}: {len(samples)} usable samples")
    if getattr(ds, "dropped", None):
        dropped = {k: v for k, v in ds.dropped.items() if v}
        if dropped:
            print(f"  dropped: {dropped}")

    cfg = ExtractionConfig(
        encoders={"text": args.text_encoder, "audio": args.audio_encoder,
                  "visual": args.visual_encoder},
        seq_len=dict(SEQ_LEN),
        audio_max_seconds=args.audio_seconds,
        layer={"audio": args.audio_layer, "text": args.text_layer},
        visual_frames=args.visual_frames,
        face_crop=args.face_crop,
    )
    fx = FeatureExtractor(cfg, modalities=args.modalities)

    if args.pilot:
        out = Path(args.out or "data/cache") / f"{args.dataset}-pilot"
        m = fx.extract(samples, out, limit=args.pilot, log_every=10)
        per = m["seconds_per_sample_steady"]
        total = per * len(samples)
        print(f"\nPILOT: {per:.3f} s/sample steady-state over {args.pilot} clips "
              f"({m['seconds_per_sample']:.3f} incl. one-off encoder load)")
        print(f"  projected full corpus ({len(samples)} samples): "
              f"{total/60:.1f} min = {total/3600:.2f} h")
        print(f"  feature cache would be roughly "
              f"{sum(v * m['dims'][k] for k, v in SEQ_LEN.items() if k in m['dims']) * 2 * len(samples) / 1e9:.2f} GB")
        return 0

    out = args.out or Path("data/cache") / args.dataset
    fx.extract(samples, out, limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
