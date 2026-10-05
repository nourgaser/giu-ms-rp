"""
label_provenance.py — how much of the overall label a single modality's own
annotation already supplies.

Both corpora annotate each modality separately (transcript / audio / frame or
video) as well as overall. For each modality we report how often its own
annotation equals the overall label. If one modality's annotation reproduces
the overall label almost always, a model with only that modality already holds
nearly all the label information, and no fusion mechanism has headroom in which
to distinguish itself.

Computed over samples that carry all three per-modality labels, so every
percentage in a column shares one denominator. No model is trained; this reads
annotations only and runs in seconds.

    python -m experiments.label_provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                           # noqa: E402

from rp2.data.armusa import ArMuSA                           # noqa: E402
from rp2.data.egymer import EgyMer                           # noqa: E402
from rp2.evaluate import core_metrics                        # noqa: E402

MODALITIES = {"transcript": "text_label", "audio": "audio_label",
              "visual": "visual_label"}


def provenance(samples) -> dict:
    s = [x for x in samples if x.has_modality_labels]
    n = len(s)
    out = {"n": n}
    for name, attr in MODALITIES.items():
        out[f"agree_{name}"] = sum(getattr(x, attr) == x.label for x in s) / n
    out["overall_differs_from_text"] = 1 - out["agree_transcript"]
    out["n_differs_from_text"] = sum(x.text_label != x.label for x in s)
    # The ceiling a model with perfect access to the transcript label would reach.
    n_labels = max(max(x.label, x.text_label) for x in s) + 1
    out["macro_f1_of_transcript_label"] = core_metrics(
        np.array([x.label for x in s]), np.array([x.text_label for x in s]),
        n_labels)["macro_f1"]
    sarc = [x for x in s if x.meta.get("sarcasm") is not None
            and x.meta.get("sarcasm_text") is not None]
    if sarc:
        out["sarcasm_agree_transcript"] = (
            sum(x.meta["sarcasm"] == x.meta["sarcasm_text"] for x in sarc) / len(sarc))
    congruence: dict[str, int] = {}
    for x in s:
        congruence[x.congruence] = congruence.get(x.congruence, 0) + 1
    out["congruence"] = congruence
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--armusa", type=Path, default=Path("data/raw/ar-musa"))
    ap.add_argument("--egymer", type=Path, default=Path("data/raw/egy-mer"))
    ap.add_argument("--out", type=Path, default=Path("results/label_provenance.json"))
    args = ap.parse_args()

    res = {"ar-musa": provenance(ArMuSA(args.armusa).samples()),
           "egy-mer": provenance(EgyMer(args.egymer).samples())}

    print(f"{'':28}{'Ar-MuSA':>10}{'EGY-MER':>10}")
    for k in ("agree_transcript", "agree_audio", "agree_visual",
              "overall_differs_from_text"):
        print(f"{k:28}{res['ar-musa'][k]:>10.1%}{res['egy-mer'][k]:>10.1%}")
    print(f"{'n (all three labels)':28}{res['ar-musa']['n']:>10}{res['egy-mer']['n']:>10}")
    print(f"{'n differs from text':28}{res['ar-musa']['n_differs_from_text']:>10}"
          f"{res['egy-mer']['n_differs_from_text']:>10}")
    print(f"{'macro-F1 of transcript label':28}"
          f"{res['ar-musa']['macro_f1_of_transcript_label']:>10.4f}"
          f"{res['egy-mer']['macro_f1_of_transcript_label']:>10.4f}")
    if "sarcasm_agree_transcript" in res["ar-musa"]:
        print(f"{'sarcasm flag = transcript':28}"
              f"{res['ar-musa']['sarcasm_agree_transcript']:>10.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
