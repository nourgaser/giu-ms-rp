"""
base.py — the dataset-independent interface.

Everything downstream of this file (feature extraction, models, training,
evaluation) sees only `Sample` and `DatasetAdapter`. No module outside
`rp2.data` may reference a dataset by name.

That is a deliberate constraint rather than tidiness. The experiment is
specified on EGY-MER, which we do not have and may never get. Making the
corpus a plug-in means the pipeline can be built, run and reported on a
corpus we *do* have, and EGY-MER becomes a new adapter plus a config entry
whenever (or if ever) it arrives — not a rewrite.

A corpus therefore has to declare three things:

  * `labels`      — the label names, in index order. Sentiment corpora have
                    3, emotion corpora have 7. Nothing downstream assumes a
                    count; the classifier head reads `len(labels)`.
  * `samples()`   — an iterable of `Sample`.
  * `modalities`  — which of text/audio/visual it actually carries. MDER-MA,
                    for instance, has no visual stream at all, so a tri-modal
                    model cannot be run on it and must fail loudly rather
                    than silently feeding zeros to a Key/Value projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

Modality = str  # one of: "text", "audio", "visual"
ALL_MODALITIES: tuple[Modality, ...] = ("text", "audio", "visual")


@dataclass(frozen=True)
class Sample:
    """One annotated clip.

    `label` is an index into the adapter's `labels` tuple. The per-modality
    fields are optional: they exist because Ar-MuSA annotates each stream
    separately, which is what makes the congruence analysis possible. A
    corpus that only carries one overall label leaves them None, and the
    congruence-stratified report is then simply unavailable for that corpus
    rather than being faked.
    """

    uid: str
    label: int

    text: str | None = None
    audio_path: Path | None = None
    visual_path: Path | None = None

    # Per-modality labels — present only where the corpus annotates them.
    text_label: int | None = None
    audio_label: int | None = None
    visual_label: int | None = None

    # Corpus-specific extras (sarcasm flags, genre, speaker id, ...).
    meta: dict = field(default_factory=dict)

    @property
    def has_modality_labels(self) -> bool:
        return None not in (self.text_label, self.audio_label, self.visual_label)

    @property
    def congruence(self) -> str | None:
        """How many distinct labels the three modalities carry.

        "congruent"    all three annotators agreed
        "partial"      exactly two distinct values
        "conflicting"  all three differ

        This is the stratification variable for the main result. It is derived
        from the corpus's own annotations, not from anything the model does, so
        the split is fixed before any model is trained and cannot be tuned.
        """
        if not self.has_modality_labels:
            return None
        n = len({self.text_label, self.audio_label, self.visual_label})
        return {1: "congruent", 2: "partial", 3: "conflicting"}[n]

    @property
    def text_contradicts_nonverbal(self) -> bool | None:
        """Text disagrees with BOTH of the non-verbal streams.

        The sharpest available proxy for "the words say one thing and the
        delivery says another" — the case Tri-CAN's text-as-Query design is
        motivated by, and the case late fusion is least equipped for.
        """
        if not self.has_modality_labels:
            return None
        return (self.text_label != self.audio_label
                and self.text_label != self.visual_label)


@runtime_checkable
class DatasetAdapter(Protocol):
    """What a corpus must provide to be usable by the rest of the pipeline."""

    name: str
    labels: Sequence[str]
    modalities: Sequence[Modality]

    def samples(self) -> Iterable[Sample]: ...

    def is_available(self) -> bool:
        """Whether the underlying data is actually present on disk.

        Kept separate from `samples()` so the CLI can report "EGY-MER: adapter
        implemented, data not present" rather than raising deep inside a
        training loop.
        """
        ...


class DataNotAvailable(RuntimeError):
    """Raised when an adapter is asked for samples it cannot produce.

    Carries the acquisition route in the message: several corpora in this area
    are 'published' but only obtainable by emailing the authors, and a bare
    FileNotFoundError loses that information.
    """
