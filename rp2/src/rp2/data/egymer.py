"""
egymer.py — EGY-MER adapter, written against the corpus as actually delivered.

EGY-MER (Hussein et al. 2025) is the Egyptian Arabic multimodal emotion corpus
the original proposal was built around. It arrived on 2026-08-09 and is
materially different from both the published description and the placeholder
this file previously contained. The differences are recorded here because they
matter for what can be claimed.

**Scale.** The paper describes 363 samples. The delivered corpus holds 5,471
segments with media, of which **505 carry a complete annotation** (three
annotators x three modalities, with a documented majority-vote resolution).
The 505 span 49 source videos, 70 minutes, mean segment length 8.3 s.

**Structure.** Not a flat table. Each source video is a directory
`videoplayback (N)/` holding speaker-diarised segments:

    videoplayback (21)/
        audios/SPEAKER_00/SPEAKER_00_segment_0000.wav
        videos/SPEAKER_00/SPEAKER_00_segment_0000.mp4

and a single spreadsheet at the top level indexes every segment across every
folder. `Folder` + `audio_file` / `video_file` give the path.

**Why this corpus can do what Ar-MuSA could not.** Each segment is annotated
*separately for text, audio and video*, and the overall label is the majority
across those three. Measuring how much of the overall label each modality alone
supplies gives:

    modality      EGY-MER (emotion)     Ar-MuSA (sentiment)
    transcript          60.4%                 96.5%
    audio               91.2%                 77.1%
    video               83.0%                 48.2%

The ordering is inverted. On Ar-MuSA the label is essentially the transcript
label, which leaves no headroom for a fusion mechanism to distinguish itself
(and none did). On EGY-MER the label lives in the *delivery*: 40.6% of
segments carry an overall label that differs from the text label, against 3.5%
on Ar-MuSA. This is the corpus property the whole RP2 argument predicted was
necessary, measured rather than assumed.

**Labels.** Seven emotions plus an `Ambiguous` category (spelled "Amiguous" in
the released file; normalised here). By default segments whose *overall* label
is Ambiguous are dropped, giving the 7-class problem the paper describes;
per-modality Ambiguous values are retained, because a modality being
uninformative is itself part of the congruence signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import DataNotAvailable, Sample

# The paper's seven emotions, in a fixed order so label indices are stable.
LABELS: tuple[str, ...] = (
    "neutral", "happiness", "sadness", "anger", "fear", "disgust", "surprise",
)
AMBIGUOUS = "ambiguous"

INDEX = Path("_meta") / "all_segments.xlsx"
MEDIA = Path("media")

# Columns in the released workbook.
COL_OVERALL = "Final Overall (majority of modalities)"
COL_TEXT = "Text Emotion (final)"
COL_AUDIO = "Emotion Audio (final)"
COL_VIDEO = "Video Emotion (final)"
COL_CLARITY = "Audio Clarity (final)"
COL_VAD = "Text VAD (final)"

ACQUISITION = """\
EGY-MER is not a public download; it was obtained from the corpus holders.
Expected layout under --root:
    _meta/all_segments.xlsx        the index (every segment, all annotations)
    media/<Folder>/audios/...wav   segment audio
    media/<Folder>/videos/...mp4   segment video
Fetch with rclone using _meta/files.txt (see rp2/README.md).
"""


def _norm(v) -> str | None:
    """Normalise a label cell. Returns None for blanks.

    'Amiguous' is a typo in the released workbook and is mapped rather than
    corrected upstream, so re-downloading the corpus does not silently change
    the label space.
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in {"nan", "none", "-"}:
        return None
    if s in {"amiguous", "ambigous", "ambiguous"}:
        return AMBIGUOUS
    aliases = {
        "happy": "happiness", "joy": "happiness",
        "sad": "sadness", "angry": "anger", "anger ": "anger",
        "fearful": "fear", "afraid": "fear",
        "disgusted": "disgust", "surprised": "surprise",
        "neutre": "neutral", "netural": "neutral",
    }
    return aliases.get(s, s)


class EgyMer:
    """Adapter for EGY-MER.

    `root` is the directory holding `_meta/` and `media/`.
    """

    name = "egy-mer"
    labels = LABELS
    modalities = ("text", "audio", "visual")
    task = "emotion"

    def __init__(self, root: str | Path,
                 require_media: bool | tuple[str, ...] = True,
                 keep_ambiguous: bool = False) -> None:
        self.root = Path(root)
        if require_media is True:
            self.require: tuple[str, ...] = ("audio", "visual")
        elif require_media is False:
            self.require = ()
        else:
            self.require = tuple(require_media)
        # When True, Ambiguous becomes an 8th class. Default False gives the
        # 7-class problem the paper describes.
        self.keep_ambiguous = keep_ambiguous
        if keep_ambiguous:
            self.labels = LABELS + (AMBIGUOUS,)
        self.dropped: dict[str, int] = {}

    # -- availability -----------------------------------------------------

    def is_available(self) -> bool:
        return (self.root / INDEX).exists()

    def has_media(self) -> bool:
        d = self.root / MEDIA
        return d.is_dir() and any(d.rglob("*.wav"))

    # -- samples ----------------------------------------------------------

    def samples(self) -> Iterable[Sample]:
        if not self.is_available():
            raise DataNotAvailable(
                f"EGY-MER index not found at {self.root / INDEX}.\n{ACQUISITION}")

        import pandas as pd

        df = pd.read_excel(self.root / INDEX)
        index = {name: i for i, name in enumerate(self.labels)}
        self.dropped = {"no_overall": 0, "ambiguous_overall": 0,
                        "unknown_label": 0, "missing_audio": 0,
                        "missing_visual": 0}

        for _, row in df.iterrows():
            overall = _norm(row.get(COL_OVERALL))
            if overall is None:
                self.dropped["no_overall"] += 1
                continue
            if overall == AMBIGUOUS and not self.keep_ambiguous:
                self.dropped["ambiguous_overall"] += 1
                continue
            if overall not in index:
                # A label outside the declared space is a schema mismatch.
                # Dropping it silently would corrupt the class distribution.
                self.dropped["unknown_label"] += 1
                continue

            folder = str(row.get("Folder") or "").strip()
            audio = self.root / MEDIA / folder / str(row.get("audio_file") or "").strip()
            visual = self.root / MEDIA / folder / str(row.get("video_file") or "").strip()

            if "audio" in self.require and not audio.exists():
                self.dropped["missing_audio"] += 1
                continue
            if "visual" in self.require and not visual.exists():
                self.dropped["missing_visual"] += 1
                continue

            # Per-modality labels are indexed in an EXTENDED space that keeps
            # Ambiguous as its own value. They are never a training target —
            # they exist only to compute congruence — and treating "this
            # modality was uninformative" as missing rather than as a distinct
            # state would silently drop 65 of 484 segments from the very
            # analysis this corpus was obtained for.
            def lbl(col: str) -> int | None:
                v = _norm(row.get(col))
                if v is None:
                    return None
                if v == AMBIGUOUS:
                    return len(LABELS)      # one past the 7 emotions
                return index.get(v)

            text = str(row.get("transcript") or "").strip() or None

            yield Sample(
                uid=f"{folder}:{row.get('speaker')}:{row.get('speaker_segment_id')}",
                label=index[overall],
                text=text,
                audio_path=audio,
                visual_path=visual,
                # These are what make the congruence analysis possible, and are
                # the reason this corpus can test fusion where Ar-MuSA cannot.
                text_label=lbl(COL_TEXT),
                audio_label=lbl(COL_AUDIO),
                visual_label=lbl(COL_VIDEO),
                meta={
                    "corpus": "egy-mer",
                    "folder": folder,
                    "speaker": str(row.get("speaker") or ""),
                    "duration": float(row.get("duration") or 0.0),
                    "clarity": _norm(row.get(COL_CLARITY)),
                    "vad": _norm(row.get(COL_VAD)),
                    # Raw per-modality names, kept so an Ambiguous modality is
                    # still visible after the numeric mapping drops it.
                    "text_emotion_raw": _norm(row.get(COL_TEXT)),
                    "audio_emotion_raw": _norm(row.get(COL_AUDIO)),
                    "video_emotion_raw": _norm(row.get(COL_VIDEO)),
                },
            )
