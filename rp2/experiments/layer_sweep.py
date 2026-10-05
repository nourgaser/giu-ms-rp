"""
layer_sweep.py — are we reading the encoders at the wrong depth?

Every result so far reads `last_hidden_state`, i.e. the final transformer
layer. For audio that is very likely the wrong choice, and it may be the real
reason the non-verbal modalities looked useless.

In wav2vec2-style models the final layers specialise toward the self-supervised
pretraining objective (predicting masked acoustic units), which discards
exactly the slowly-varying paralinguistic information -- tone, affect, speaker
state -- that a sentiment or emotion task needs. Probing studies consistently
find that information peaks in the MIDDLE of the stack. The same effect is
milder but real for BERT-style text encoders.

If that is what happened, then two earlier conclusions are wrong:
  * "audio contributes almost nothing" (audio-only macro-F1 0.42), and
  * "a multilingual encoder does not help" (XLS-R scored WORSE than the
    English model) -- because both were read at the wrong depth, so the
    comparison never tested what it claimed to.

This sweeps the layer index for one modality, holding everything else fixed.

    python -m experiments.layer_sweep --modality audio --layers 4 6 8 10 12
    python -m experiments.layer_sweep --modality text  --layers 6 8 10 12
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2.data import get_dataset                               # noqa: E402
from rp2.evaluate import core_metrics                          # noqa: E402
from rp2.features import (DEFAULT_ENCODERS, ExtractionConfig,   # noqa: E402
                          FeatureExtractor, SEQ_LEN, load_cache)
from rp2.train import TrainConfig, cross_validate               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=("text", "audio"))
    ap.add_argument("--dataset", default="ar-musa")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache/layers"))
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 6, 8, 10, 12])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    encoder = args.encoder or DEFAULT_ENCODERS[args.modality]
    root = args.root or Path(f'data/raw/{args.dataset}')
    samples = list(get_dataset(args.dataset, root).samples())[: args.n]
    print(f"{args.modality}: {encoder}")
    print(f"sweeping layers {args.layers} (-1 = final) on {len(samples)} samples\n")

    rows = []
    for layer in args.layers + [-1]:
        out = args.cache_root / f"{args.modality}-L{layer}"
        print(f"--- layer {layer}")
        try:
            if not (out / "manifest.json").exists():
                cfg = ExtractionConfig(
                    encoders={m: (encoder if m == args.modality else "")
                              for m in ("text", "audio", "visual")},
                    seq_len=dict(SEQ_LEN),
                    layer={args.modality: layer},
                )
                FeatureExtractor(cfg, modalities=(args.modality,)).extract(
                    samples, out, log_every=1000)

            feats, masks, labels, _, _ = load_cache(out, (args.modality,))
            n_labels = int(labels.max()) + 1
            tcfg = TrainConfig(epochs=args.epochs, n_folds=args.folds, seed=0,
                               n_threads=args.threads)
            res = cross_validate("late-fusion", feats, labels, n_labels, tcfg, masks)
            m = core_metrics(labels, res["y_pred"], n_labels)
            rows.append((layer, m))
            print(f"    macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}\n",
                  flush=True)
        except Exception:
            print(f"    FAILED:\n{traceback.format_exc(limit=2)}", flush=True)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    if not rows:
        return 1

    rows.sort(key=lambda r: -r[1]["macro_f1"])
    print(f"\n{'layer':>7s} {'macro-F1':>10s} {'acc':>8s}")
    print("-" * 28)
    for layer, m in rows:
        tag = "final" if layer == -1 else str(layer)
        print(f"{tag:>7s} {m['macro_f1']:10.4f} {m['accuracy']:8.4f}")

    best = rows[0]
    final = next((m for l, m in rows if l == -1), None)
    print(f"\nbest layer: {best[0]}  macro-F1 {best[1]['macro_f1']:.4f}")
    if final is not None:
        d = best[1]["macro_f1"] - final["macro_f1"]
        print(f"  vs the final layer used so far: {d:+.4f}")
        if d > 0.03:
            print("  -> Reading the final layer was costing real signal. The earlier\n"
                  "     'this modality contributes nothing' conclusion has to be\n"
                  "     re-tested at this depth.")

    out_path = args.out or Path(f"results/layer_sweep-{args.dataset}-{args.modality}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"modality": args.modality, "encoder": encoder, "n": len(samples),
         "results": {str(l): m for l, m in rows}}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
