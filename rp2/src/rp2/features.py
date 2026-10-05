"""
features.py — frozen encoder feature extraction, cached to disk.

The encoders are never trained. Features are extracted once per corpus and
reused by every model and every fold. This is the compute protocol forced by
the hardware (CPU only — the machine's RX 5700 XT has no usable PyTorch
backend), but it is also the right protocol for the question being asked:

    With the encoders frozen and the cached features shared, the ONLY thing
    that differs between the late-fusion baseline and Tri-CAN is the fusion
    mechanism. Any difference in F1 is therefore attributable to fusion.

Fine-tuning the encoders would confound the comparison with how well each
architecture happens to propagate gradients into AraBERT. The proposal's
LoRA-on-encoders plan becomes an optional ablation, not the main result.

Sequence lengths are fixed per modality and padded with an explicit mask, so
the attention in models.py never attends to padding:

    text     48 tokens   AraBERT subwords
    audio    32 frames   mean-pooled from the encoder's own frame rate
    visual   49 tokens   Swin's final-stage 7x7 patch grid, ONE frame per clip

Storage is float16 (halves the cache; the models cast to float32 on load).
For Ar-MuSA at 8,700 samples the cache is roughly 2 GB, which matters on a
disk with ~20 GB free.

Run `rp2-extract --pilot 50` first: it times a 50-clip run and prints the
projected wall time for the full corpus, so the decision to subsample is made
on a measurement rather than a guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Defaults chosen for CPU feasibility, all overridable from the CLI.
DEFAULT_ENCODERS = {
    # Arabic BERT, the standard choice for Egyptian/MSA text.
    "text": "aubmindlab/bert-base-arabertv2",
    # Multilingual speech. base-sized: XLS-R-300m is ~4x the CPU cost for a
    # frozen feature extractor, which the pilot timing did not justify.
    "audio": "facebook/wav2vec2-base",
    # Swin-tiny over the single released frame.
    "visual": "microsoft/swin-tiny-patch4-window7-224",
}

SEQ_LEN = {"text": 48, "audio": 32, "visual": 49}
AUDIO_SR = 16_000
AUDIO_MAX_SECONDS = 8.0   # clips run to 19.9s; the tail is mostly silence


@dataclass
class ExtractionConfig:
    encoders: dict[str, str]
    seq_len: dict[str, int]
    audio_sr: int = AUDIO_SR
    audio_max_seconds: float = AUDIO_MAX_SECONDS
    dtype: str = "float16"
    # Detect the largest face in each frame and crop to it before encoding.
    #
    # This matters specifically for expression-pretrained visual encoders. They
    # are trained on tight, centred face crops (FER2013 and its descendants),
    # whereas EGY-MER frames are full talk-show shots in which the face may be
    # a small part of the image. Feeding a whole scene to a face-expression
    # model spends most of its capacity on background. Frames with no detected
    # face fall back to the full frame rather than being dropped, so the
    # sequence length stays comparable across samples.
    face_crop: bool = False
    face_margin: float = 0.35        # fraction of box size added on each side
    face_model: str = "data/models/face_detection_yunet_2023mar.onnx"
    # Frames sampled per video segment. Ignored for still-image corpora.
    # 16 over EGY-MER's mean 8.3 s segment is roughly 2 fps — enough to carry
    # expression change without making extraction the bottleneck.
    visual_frames: int = 16
    # Which transformer layer to read features from, per modality.
    # -1 = the final layer, which is what the first study used throughout.
    #
    # That was very likely a mistake for audio. In wav2vec2-style models the
    # last layers specialise toward the self-supervised pretraining objective
    # (masked unit prediction), while paralinguistic content -- tone, affect,
    # speaker state -- is best represented in the MIDDLE of the stack. Reading
    # layer -1 may be why audio-only scored 0.42, and why swapping to a
    # multilingual encoder did not help: both were read at the wrong depth.
    # `layer` makes that testable rather than assumed.
    layer: dict[str, int] = field(default_factory=dict)

    def layer_for(self, modality: str) -> int:
        return self.layer.get(modality, -1)

    @classmethod
    def default(cls) -> "ExtractionConfig":
        return cls(encoders=dict(DEFAULT_ENCODERS), seq_len=dict(SEQ_LEN))


def _pad_or_truncate(x: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a (L, D) array to (target, D), returning the array and a keep-mask."""
    L, D = x.shape
    if L >= target:
        return x[:target], np.ones(target, dtype=bool)
    out = np.zeros((target, D), dtype=x.dtype)
    out[:L] = x
    mask = np.zeros(target, dtype=bool)
    mask[:L] = True
    return out, mask


def _resample_sequence(x: np.ndarray, target: int) -> np.ndarray:
    """Mean-pool a variable-length (L, D) sequence into exactly `target` bins.

    Used for audio, where the encoder's frame count scales with clip duration.
    Binning rather than truncating keeps information from the whole clip; the
    alternative (take the first N frames) would discard the end of every long
    utterance, which is where a contradicting prosodic cue often sits.
    """
    L = x.shape[0]
    if L == 0:
        return np.zeros((target, x.shape[1]), dtype=x.dtype)
    edges = np.linspace(0, L, target + 1).astype(int)
    out = np.stack([
        x[a:b].mean(axis=0) if b > a else x[min(a, L - 1)]
        for a, b in zip(edges[:-1], edges[1:])
    ])
    return out.astype(x.dtype)


class FeatureExtractor:
    """Loads the three frozen encoders lazily and produces cached arrays."""

    def __init__(self, cfg: ExtractionConfig | None = None,
                 modalities: Sequence[str] = ("text", "audio", "visual")) -> None:
        self.cfg = cfg or ExtractionConfig.default()
        self.modalities = tuple(modalities)
        self._loaded: dict[str, tuple] = {}

    # -- lazy encoder loading ---------------------------------------------

    def _text(self):
        if "text" not in self._loaded:
            import torch
            from transformers import AutoModel, AutoTokenizer
            name = self.cfg.encoders["text"]
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self._loaded["text"] = (tok, mdl, torch)
        return self._loaded["text"]

    def _audio(self):
        if "audio" not in self._loaded:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
            name = self.cfg.encoders["audio"]
            fe = AutoFeatureExtractor.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self._loaded["audio"] = (fe, mdl, torch)
        return self._loaded["audio"]

    def _visual(self):
        if "visual" not in self._loaded:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            name = self.cfg.encoders["visual"]
            proc = AutoImageProcessor.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).eval()
            # CLIP loads as a dual text+image model that cannot be called with
            # pixel_values alone; we want only its vision tower. Same for any
            # other model exposing a `vision_model` submodule.
            if hasattr(mdl, "vision_model"):
                mdl = mdl.vision_model
            mdl = mdl.eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self._loaded["visual"] = (proc, mdl, torch)
        return self._loaded["visual"]

    # -- per-sample encoding ----------------------------------------------

    def encode_text(self, text: str | None) -> tuple[np.ndarray, np.ndarray]:
        tok, mdl, torch = self._text()
        target = self.cfg.seq_len["text"]
        if not text:
            d = mdl.config.hidden_size
            return np.zeros((target, d), np.float32), np.zeros(target, bool)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=target)
        layer = self.cfg.layer_for("text")
        with torch.no_grad():
            if layer == -1:
                h = mdl(**enc).last_hidden_state[0].numpy()
            else:
                out = mdl(**enc, output_hidden_states=True)
                h = out.hidden_states[layer][0].numpy()
        return _pad_or_truncate(h, target)

    def encode_audio(self, path: Path | None) -> tuple[np.ndarray, np.ndarray]:
        fe, mdl, torch = self._audio()
        target = self.cfg.seq_len["audio"]
        d = mdl.config.hidden_size
        if path is None or not Path(path).exists():
            return np.zeros((target, d), np.float32), np.zeros(target, bool)

        import librosa
        wav, _ = librosa.load(str(path), sr=self.cfg.audio_sr, mono=True)
        wav = wav[: int(self.cfg.audio_sr * self.cfg.audio_max_seconds)]
        if wav.size < self.cfg.audio_sr // 10:      # under 100 ms: unusable
            return np.zeros((target, d), np.float32), np.zeros(target, bool)

        inp = fe(wav, sampling_rate=self.cfg.audio_sr, return_tensors="pt")
        layer = self.cfg.layer_for("audio")
        with torch.no_grad():
            if layer == -1:
                h = mdl(**inp).last_hidden_state[0].numpy()
            else:
                out = mdl(**inp, output_hidden_states=True)
                h = out.hidden_states[layer][0].numpy()
        return _resample_sequence(h, target), np.ones(target, bool)

    def encode_visual(self, path: Path | None) -> tuple[np.ndarray, np.ndarray]:
        """Encode the visual stream — a still image or a video segment.

        The two corpora differ here, and the difference is worth stating rather
        than hiding behind a shared function name:

          Ar-MuSA ships ONE frame per clip. The token sequence is then Swin's
          SPATIAL patch grid, and attention over it is attention over image
          regions, not over time.

          EGY-MER ships the actual video segment. Frames are sampled uniformly,
          each is encoded and pooled to one vector, and the sequence is
          TEMPORAL. This is the arrangement cross-modal attention was designed
          for, and it is the first time in this project that Tri-CAN can attend
          over time at all.
        """
        proc, mdl, torch = self._visual()
        target = self.cfg.seq_len["visual"]
        d = getattr(mdl.config, "hidden_size", None) or 768
        if path is None or not Path(path).exists():
            return np.zeros((target, d), np.float32), np.zeros(target, bool)

        from PIL import Image

        if Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            frames = self._video_frames(Path(path), self.cfg.visual_frames)
            if not frames:
                return np.zeros((target, d), np.float32), np.zeros(target, bool)
            inp = proc(images=frames, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**inp).last_hidden_state       # (n_frames, patches, D)
            # One vector per frame: the sequence axis becomes time.
            h = out.mean(dim=1).numpy()
            return _pad_or_truncate(h, target)

        img = Image.open(path).convert("RGB")
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            h = mdl(**inp).last_hidden_state[0].numpy()
        return _pad_or_truncate(h, target)

    def _video_frames(self, path: Path, n: int) -> list:
        """Sample `n` frames uniformly from a video via ffmpeg.

        ffmpeg rather than a Python decoder because it is already installed,
        handles every container in this corpus, and keeps the dependency list
        short. Frames go to a temp directory and are removed immediately.
        """
        import shutil
        import subprocess
        import tempfile

        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="rp2frames-"))
        try:
            dur = self._duration(path)
            if dur <= 0:
                return []
            # ONE decode pass rather than n seeks. Seeking n times re-opens and
            # re-decodes the file each time; at 16 frames per segment over 484
            # segments that is ~7,700 process launches for no benefit.
            # `fps=n/dur` samples evenly across the whole segment.
            fps = max(n / dur, 0.01)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                 "-vf", f"fps={fps:.6f},scale=256:-1", "-frames:v", str(n),
                 "-q:v", "3", str(tmp / "f%04d.jpg")],
                check=False, timeout=180,
            )
            files = sorted(tmp.glob("f*.jpg"))
            frames = [Image.open(f).convert("RGB") for f in files
                      if f.stat().st_size > 0]
            if not frames:
                return []
            if self.cfg.face_crop:
                frames = [self._crop_face(f) for f in frames]
            # Short or oddly-timed segments can yield fewer than n frames;
            # repeat the last so every sample has a comparable sequence rather
            # than silently varying in length.
            while len(frames) < n:
                frames.append(frames[-1])
            return frames[:n]
        except Exception:
            return []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _crop_face(self, img):
        """Crop to the largest detected face, or return the frame unchanged.

        Uses OpenCV's YuNet detector. The obvious cheap choice would have been
        a Haar cascade, but OpenCV 5 removed `CascadeClassifier` from the
        Python bindings; YuNet is the supported replacement, is a small ONNX
        model (230 KB), runs comfortably on CPU at the 16-frames-per-segment
        rate this needs, and is markedly more accurate on non-frontal faces —
        which matters for talk-show footage where speakers turn away.

        A missed detection returns the whole frame, i.e. exactly the behaviour
        before cropping existed, so failure can never be worse than not
        cropping. `faces_detected` / `frames_seen` record the hit rate so the
        write-up can state how often cropping actually applied instead of
        assuming it did.
        """
        try:
            import cv2
            import numpy as _np
        except Exception:
            return img

        model = self.cfg.face_model
        if not model or not Path(model).exists():
            return img

        self.frames_seen = getattr(self, "frames_seen", 0) + 1
        arr = _np.array(img)[:, :, ::-1]          # PIL RGB -> OpenCV BGR
        H, W = arr.shape[:2]

        det = getattr(self, "_yunet", None)
        if det is None:
            det = cv2.FaceDetectorYN_create(str(model), "", (W, H), 0.6, 0.3, 5000)
            self._yunet = det
        det.setInputSize((W, H))

        try:
            _, faces = det.detect(arr)
        except Exception:
            return img
        if faces is None or len(faces) == 0:
            return img

        # YuNet rows are [x, y, w, h, 5 landmarks..., score]
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
        m = self.cfg.face_margin
        x0, y0 = max(int(x - m * w), 0), max(int(y - m * h), 0)
        x1, y1 = min(int(x + w + m * w), W), min(int(y + h + m * h), H)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return img
        self.faces_detected = getattr(self, "faces_detected", 0) + 1
        return img.crop((x0, y0, x1, y1))

    @staticmethod
    def _duration(path: Path) -> float:
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip() or 0.0)
        except Exception:
            return 0.0

    # -- corpus-level extraction ------------------------------------------

    def extract(self, samples: Iterable, out_dir: Path,
                limit: int | None = None, log_every: int = 100) -> dict:
        """Encode every sample, writing arrays and a manifest to `out_dir`.

        Writes one .npy per modality plus masks, plus manifest.json recording
        the encoder names and sequence lengths. Downstream code refuses to load
        a cache whose manifest does not match the requested configuration, so a
        stale cache cannot silently contaminate a run.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        samples = list(samples)
        if limit:
            samples = samples[:limit]
        n = len(samples)
        if n == 0:
            raise ValueError("no samples to extract")

        feats: dict[str, list] = {m: [] for m in self.modalities}
        masks: dict[str, list] = {m: [] for m in self.modalities}
        started = time.time()

        # The encoders load lazily on the first sample, which on this machine
        # costs ~110 s. Averaging that over a 40-clip pilot inflates the
        # per-sample figure roughly sixfold and would lead straight to a wrong
        # decision about subsampling. So warm-up is timed separately and the
        # steady-state rate is what gets reported and extrapolated.
        WARMUP = 5
        warm_end = None

        for i, s in enumerate(samples):
            if "text" in self.modalities:
                f, m = self.encode_text(s.text)
                feats["text"].append(f.astype(np.float16)); masks["text"].append(m)
            if "audio" in self.modalities:
                f, m = self.encode_audio(s.audio_path)
                feats["audio"].append(f.astype(np.float16)); masks["audio"].append(m)
            if "visual" in self.modalities:
                f, m = self.encode_visual(s.visual_path)
                feats["visual"].append(f.astype(np.float16)); masks["visual"].append(m)

            if i + 1 == WARMUP:
                warm_end = time.time()

            if log_every and (i + 1) % log_every == 0:
                el = time.time() - started
                rate = ((time.time() - warm_end) / (i + 1 - WARMUP)
                        if warm_end and i + 1 > WARMUP else el / (i + 1))
                print(f"  {i+1}/{n}  {el:6.1f}s  ({rate:.3f} s/sample steady, "
                      f"eta {(n-i-1)*rate/60:.1f} min)", flush=True)

        elapsed = time.time() - started
        steady = ((time.time() - warm_end) / max(n - WARMUP, 1)
                  if warm_end and n > WARMUP else elapsed / n)
        for m in self.modalities:
            np.save(out_dir / f"{m}.npy", np.stack(feats[m]))
            np.save(out_dir / f"{m}_mask.npy", np.stack(masks[m]))

        np.save(out_dir / "labels.npy", np.array([s.label for s in samples], dtype=int))
        meta = {
            "uid": [s.uid for s in samples],
            "congruence": [s.congruence for s in samples],
            "sarcasm": [s.meta.get("sarcasm") for s in samples],
            "text_label": [s.text_label for s in samples],
            "audio_label": [s.audio_label for s in samples],
            "visual_label": [s.visual_label for s in samples],
        }
        (out_dir / "meta.json").write_text(json.dumps(meta))

        manifest = {
            "n": n,
            "modalities": list(self.modalities),
            "config": asdict(self.cfg),
            "dims": {m: int(np.stack(feats[m]).shape[-1]) for m in self.modalities},
            "seconds": elapsed,
            # Wall-clock average, including the one-off encoder load.
            "seconds_per_sample": elapsed / n,
            # What to extrapolate from: excludes encoder loading.
            "seconds_per_sample_steady": steady,
            "warmup_samples": WARMUP,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nextracted {n} samples in {elapsed/60:.1f} min "
              f"({elapsed/n:.3f} s/sample incl. warm-up, "
              f"{steady:.3f} s/sample steady) -> {out_dir}")
        return manifest


def load_cache(cache_dir: Path, modalities: Sequence[str] | None = None):
    """Load a feature cache. Returns (feats, masks, labels, meta, manifest)."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    mods = tuple(modalities or manifest["modalities"])

    missing = [m for m in mods if m not in manifest["modalities"]]
    if missing:
        raise KeyError(f"cache at {cache_dir} has no features for {missing}")

    feats = {m: np.load(cache_dir / f"{m}.npy").astype(np.float32) for m in mods}
    masks = {m: np.load(cache_dir / f"{m}_mask.npy") for m in mods}
    labels = np.load(cache_dir / "labels.npy")
    meta = json.loads((cache_dir / "meta.json").read_text())
    return feats, masks, labels, meta, manifest
