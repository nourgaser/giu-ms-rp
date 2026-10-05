"""
armusa.py — Ar-MuSA adapter.

Ar-MuSA (Khaled et al. 2025, IJIES 18(4) 30-44, doi:10.22266/ijies2025.0531.03)
is an 8,700-sample Arabic multimodal sentiment corpus. It is openly downloadable
from the Hugging Face hub at `Skhaled/Ar-MUSA` under AFL-3.0, research use only.
No access request, no gate. That availability is the reason it is the primary
corpus here (see ../../PLAN.md §3).

Two properties of the corpus that the experiment depends on, both verified
against the released annotation CSVs rather than taken from the paper:

1.  Every one of the 8,700 rows carries FOUR sentiment labels — one per
    modality (audio / frame / transcript) plus an overall label — and four
    matching sarcasm flags. The per-modality labels are what make the
    congruence-stratified analysis possible at all.

2.  The visual stream is a SINGLE extracted frame per clip, not a video.
    This is easy to miss: the directory is called `frames/` and there is
    exactly one .jpg per sample ID. It has two consequences. Feature
    extraction is cheap (one Swin forward per clip, not one per frame, which
    is what makes the whole study CPU-feasible), and the visual token
    sequence Tri-CAN attends over is SPATIAL — Swin's patch grid — not
    temporal. The model is written for that and says so.

Layout on the hub, thirteen batches named <number><team-letter>:

    Annotation1a/
        updated_annotation_1a.csv
        audio_segments/audio_<id>_1a.wav
        frames/frame_<id>_1a.jpg
        transcripts/transcript_<id>_1a.txt

The CSV carries the transcript text inline, so the `transcripts/` directory is
not read — there are only 8,205 .txt files against 8,700 rows, and using the
CSV column avoids silently dropping the 495 rows whose .txt is missing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .base import DataNotAvailable, Sample

# Label 0/1/2 as released. The CSV header is literally "[0|1|2]" and the paper
# describes a three-way negative/neutral/positive scheme.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

BATCHES: tuple[str, ...] = (
    "1a", "1b", "2a", "2b", "3a", "3b", "3c",
    "4a", "4b", "4c", "5a", "5b", "5c",
)

HF_REPO = "Skhaled/Ar-MUSA"

_SENT = "{} Sentiment [0|1|2]"
_SARC = "{} Sarcasm [0|1]"


def _as_label(raw: str | None) -> int | None:
    """Parse a label cell, returning None for anything not in {0,1,2}.

    The released CSVs contain a small number of malformed cells (two empty
    audio-sarcasm values and a stray '2' and '9' in binary sarcasm columns).
    They are dropped rather than guessed at; `ArMuSA.samples()` counts them.
    """
    v = (raw or "").strip()
    return int(v) if v in {"0", "1", "2"} else None


def _as_flag(raw: str | None) -> int | None:
    v = (raw or "").strip()
    return int(v) if v in {"0", "1"} else None


class ArMuSA:
    """Adapter for Ar-MuSA. `root` is the directory holding Annotation*/."""

    name = "ar-musa"
    labels = LABELS
    modalities = ("text", "audio", "visual")
    task = "sentiment"

    def __init__(self, root: str | Path,
                 require_media: bool | Sequence[str] = True) -> None:
        self.root = Path(root)
        # True  -> every media file must exist
        # False -> yield regardless; used by the label-only analyses (congruence
        #          statistics, splits), which run before the 10 GB download
        # ("audio",) etc. -> require only the modalities actually being encoded,
        #          so a text+audio run is not blocked by absent frames
        if require_media is True:
            self.require: tuple[str, ...] = ("audio", "visual")
        elif require_media is False:
            self.require = ()
        else:
            self.require = tuple(require_media)
        self.dropped: dict[str, int] = {}

    # -- availability -----------------------------------------------------

    def is_available(self) -> bool:
        return any((self.root / f"Annotation{b}").is_dir() for b in BATCHES)

    def has_media(self) -> bool:
        for b in BATCHES:
            d = self.root / f"Annotation{b}" / "audio_segments"
            if d.is_dir() and any(d.glob("*.wav")):
                return True
        return False

    def available_batches(self) -> list[str]:
        return [b for b in BATCHES if (self.root / f"Annotation{b}").is_dir()]

    # -- samples ----------------------------------------------------------

    def samples(self) -> Iterable[Sample]:
        if not self.is_available():
            raise DataNotAvailable(
                f"Ar-MuSA not found under {self.root}.\n"
                f"It is an open download — no access request needed:\n"
                f"    rp2-fetch armusa --root {self.root}\n"
                f"or manually from https://huggingface.co/datasets/{HF_REPO}"
            )

        self.dropped = {"bad_label": 0, "missing_audio": 0, "missing_visual": 0}

        for batch in self.available_batches():
            bdir = self.root / f"Annotation{batch}"
            csv_path = bdir / f"updated_annotation_{batch}.csv"
            if not csv_path.exists():
                continue

            with csv_path.open(encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    s = self._row_to_sample(row, bdir, batch)
                    if s is not None:
                        yield s

    def _row_to_sample(self, row: dict, bdir: Path, batch: str) -> Sample | None:
        overall = _as_label(row.get(_SENT.format("Overall")))
        if overall is None:
            self.dropped["bad_label"] += 1
            return None

        audio = bdir / (row.get("Audio") or "").strip()
        visual = bdir / (row.get("Frame") or "").strip()

        if "audio" in self.require and not audio.exists():
            self.dropped["missing_audio"] += 1
            return None
        if "visual" in self.require and not visual.exists():
            self.dropped["missing_visual"] += 1
            return None

        return Sample(
            uid=f"{batch}:{(row.get('ID') or '').strip()}",
            label=overall,
            text=(row.get("Transcript") or "").strip() or None,
            audio_path=audio,
            visual_path=visual,
            text_label=_as_label(row.get(_SENT.format("Transcript"))),
            audio_label=_as_label(row.get(_SENT.format("Audio"))),
            visual_label=_as_label(row.get(_SENT.format("Frame"))),
            meta={
                "batch": batch,
                "category": (row.get("Category") or "").strip(),
                "sarcasm": _as_flag(row.get(_SARC.format("Overall"))),
                "sarcasm_text": _as_flag(row.get(_SARC.format("Transcript"))),
                "sarcasm_audio": _as_flag(row.get(_SARC.format("Audio"))),
                "sarcasm_visual": _as_flag(row.get(_SARC.format("Frame"))),
            },
        )
