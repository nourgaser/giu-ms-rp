"""
synthetic.py — a corpus made of noise, for testing the pipeline.

Not a scientific artefact. It exists so the models, training loop, metrics and
congruence report can be exercised end-to-end without any real data — which is
how the pipeline was built while EGY-MER was unavailable and before Ar-MuSA had
finished downloading.

It also serves as a *control on the analysis itself*. The main result claims
that cross-modal attention helps more on incongruent samples. Run the same
analysis on `SyntheticCorpus(signal=0.0)`, where by construction no such
structure exists, and the effect must vanish. If a null corpus produces the
headline effect, the effect is an artefact of the metric or the split, not of
the model. `tests/test_null_control.py` asserts exactly that.

Feature tensors are generated directly, bypassing the encoders, so this runs in
seconds on any machine.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import Sample


class SyntheticCorpus:
    """Configurable fake corpus with a controllable amount of real signal.

    `signal` interpolates between two regimes:
      0.0  labels are independent of the features — nothing is learnable, and
           every model must sit at chance.
      1.0  the label is cleanly recoverable from the modality features.

    `incongruent_frac` controls how often the three per-modality labels
    disagree, so the stratified report has both strata populated.
    """

    name = "synthetic"
    task = "synthetic"
    modalities = ("text", "audio", "visual")

    def __init__(
        self,
        n: int = 600,
        n_labels: int = 3,
        signal: float = 0.7,
        incongruent_frac: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.labels = tuple(f"class{i}" for i in range(n_labels))
        self.signal = float(signal)
        self.incongruent_frac = float(incongruent_frac)
        self.seed = seed

    def is_available(self) -> bool:
        return True

    def samples(self) -> Iterable[Sample]:
        """Generate modality labels first, then derive the overall label.

        This ordering matters. An earlier version set the text label equal to
        the overall label unconditionally, which made text a perfect predictor
        and drove every model to macro-F1 1.0 — the corpus could not
        distinguish a good model from a bad one, let alone detect the
        congruence effect it exists to test.

        Here the overall label is the MAJORITY of the three modality labels,
        and when all three disagree the audio label wins ("delivery decides").
        Both rules mirror how Ar-MuSA's overall annotation behaves, and both
        mean that on incongruent samples text alone is genuinely insufficient
        — which is the phenomenon under study.
        """
        rng = np.random.default_rng(self.seed)
        k = len(self.labels)

        for i in range(self.n):
            base = int(rng.integers(k))
            others = [c for c in range(k) if c != base]

            if rng.random() < self.incongruent_frac:
                t = base
                a = int(rng.choice(others))
                v = base if rng.random() < 0.5 else int(rng.choice(others))
            else:
                t = a = v = base

            counts = np.bincount([t, a, v], minlength=k)
            y = int(counts.argmax()) if counts.max() > 1 else a

            yield Sample(
                uid=f"syn{i:05d}",
                label=y,
                text=f"synthetic utterance {i}",
                text_label=t,
                audio_label=a,
                visual_label=v,
                meta={"synthetic": True, "sarcasm": int(t != a)},
            )

    # -- feature generation ------------------------------------------------

    def features(self, dims: dict[str, int]) -> dict[str, np.ndarray]:
        """Produce (n, seq, dim) float32 arrays keyed by modality.

        Each modality's features are a class-dependent mean plus noise, mixed
        by `signal`. At signal=0 the class mean contributes nothing.
        """
        rng = np.random.default_rng(self.seed + 1)
        samples = list(self.samples())
        k = len(self.labels)
        out: dict[str, np.ndarray] = {}

        seq_len = {"text": 16, "audio": 24, "visual": 49}

        for mod, dim in dims.items():
            centres = rng.normal(0, 1, size=(k, dim)).astype(np.float32)
            L = seq_len.get(mod, 16)
            arr = rng.normal(0, 1, size=(len(samples), L, dim)).astype(np.float32)

            for i, s in enumerate(samples):
                # Each modality keys off its OWN label, so that incongruent
                # samples genuinely carry conflicting evidence — which is what
                # the stratified analysis is supposed to detect.
                lbl = {"text": s.text_label, "audio": s.audio_label,
                       "visual": s.visual_label}.get(mod, s.label)
                arr[i] += self.signal * centres[lbl if lbl is not None else s.label]

            out[mod] = arr

        return out
