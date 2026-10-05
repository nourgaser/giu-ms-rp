# RP2 — Does cross-modal attention still help when the modalities disagree?

**Short version.** The experiment RP1 proposed was specified on EGY-MER, which we do not
have and may never get. This code is built so that the study runs *now*, on a corpus that is
openly downloadable, and so that EGY-MER becomes a one-line change if it ever arrives.

---

## The question

Cross-modal attention is supposed to let one modality condition on another. If that is what
it is really doing, its advantage over late fusion ought to be **largest on the samples where
the modalities carry conflicting evidence**, and small where they already agree.

Nobody has tested this. The three 2026 papers competing on Ar-MuSA report aggregate F1 only.

> **Does the advantage of cross-modal attention over late fusion depend on whether the
> modalities agree with each other?**

## Why this question and not "beat the benchmark"

Ar-MuSA already has three 2026 papers on it, the best reporting F1 0.8989. Entering that race
with a frozen-encoder CPU model would place fourth and prove nothing.

The congruence contrast is a **within-study** comparison, which means:

- it does not depend on absolute performance — if both models are mediocre, the *difference
  between strata* is still meaningful;
- it is directly implied by the architecture, so a null result is informative rather than
  embarrassing: it would be evidence against the "attention resolves conflicting evidence"
  story, and gets reported as such;
- the stratification comes from the corpus's own per-modality annotations, fixed before any
  model is trained, so it cannot be tuned to flatter the result.

**Both outcomes are a complete RP2.** That is deliberate.

## What makes it possible

Ar-MuSA annotates sentiment **separately for transcript, audio and frame**, plus an overall
label, plus sarcasm flags for each. Verified against the released CSVs (all 8,700 rows):

| | share |
|---|---|
| all three modalities agree (**congruent**) | 39.0% |
| exactly two distinct labels (**partial**) | 56.3% |
| all three differ (**conflicting**) | 4.7% |
| **incongruent** (any disagreement) | **61.0%** |
| flagged sarcastic | 13.3% |
| text disagrees with *both* audio and frame | 15.1% |

So "the modalities disagree" is an observed property of the data, not something we infer.

## Design

|  | role |
|---|---|
| **Ar-MuSA** — 8,700 samples, 3-class sentiment, open download | primary; where the study runs today |
| **EGY-MER** — 363 samples, 7-class emotion, not obtainable | target; adapter written, drops in if it arrives |
| **MORAD** — 2,325 videos, same 7 emotions, email-request only | control, if granted |

Three models, identical features, identical folds, identical training loop:

| model | fusion |
|---|---|
| `late-fusion` | pool each modality, concatenate, classify — what the published baselines do |
| `tri-can` | text queries audio and visual before pooling |
| `late-fusion-wide` | **capacity-matched control** — late fusion widened to Tri-CAN's parameter count |

The third exists so that "isn't Tri-CAN just bigger?" has an answer. If Tri-CAN beats plain
late fusion but not the capacity-matched one, the gain was parameters, not attention.

**Headline statistic:** difference-in-differences on macro-F1 —

```
DiD = [F1(TriCAN) - F1(LateFusion)] on incongruent
    - [F1(TriCAN) - F1(LateFusion)] on congruent
```

with a paired bootstrap CI (both models resampled with the same indices, since they saw the
same folds and the same samples).

## Two honest caveats, stated up front

1. **Ar-MuSA's visual modality is a single frame per clip**, not a video. Easy to miss — the
   folder is called `frames/` and there is exactly one `.jpg` per sample. So Tri-CAN's visual
   attention is over Swin's *spatial* patch grid, not a temporal sequence. It must not be
   described as temporal.
2. **This is sentiment, not emotion.** The task-regime question RP1 raises (does the result
   survive the move to 7-class emotion on hundreds of samples?) is exactly what EGY-MER or
   MORAD would answer, and is left open until one of them is obtainable.

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio torchvision
uv pip install --python .venv/bin/python \
    transformers scikit-learn pandas numpy pillow soundfile librosa huggingface_hub pytest \
    openpyxl opencv-python-headless audioread
```

```bash
# 1. prove the pipeline works — no data required, ~1 minute
.venv/bin/python -m experiments.run --dataset synthetic

# 2. the null control: no signal must produce no skill and no DiD effect
.venv/bin/python -m pytest tests -m slow

# 3. get the corpus (~10 GB, resumable, skips files we never read)
.venv/bin/python -m experiments.fetch_armusa --root data/raw/ar-musa

# 4. time it before committing to it
.venv/bin/python -m experiments.extract --dataset ar-musa --root data/raw/ar-musa --pilot 50

# 5. extract, then run
.venv/bin/python -m experiments.extract --dataset ar-musa --root data/raw/ar-musa \
    --out data/cache/ar-musa
.venv/bin/python -m experiments.run --dataset ar-musa --cache data/cache/ar-musa
```

If EGY-MER arrives: fill in the layout in `src/rp2/data/egymer.py` (the label space is
already correct; only the on-disk arrangement is assumed), then

```bash
.venv/bin/python -m experiments.extract --dataset egy-mer --root <path> --out data/cache/egy-mer
.venv/bin/python -m experiments.run --dataset egy-mer --cache data/cache/egy-mer
```

Nothing else changes. The classifier head reads `len(labels)`, so 7 classes need no edit, and
the congruence analysis switches itself off if the corpus has no per-modality labels rather
than fabricating a stratum.

## Protocol

**CPU-only, frozen encoders, features extracted once and cached.** Forced by the hardware
(the machine's RX 5700 XT has no usable PyTorch backend), but it is also the right protocol:
with the encoders frozen and the features shared, the *only* difference between the baseline
and Tri-CAN is the fusion mechanism, so any difference in F1 is attributable to it.
Fine-tuning the encoders would confound the comparison with how well each architecture
propagates gradients into AraBERT. LoRA becomes an optional ablation, not the main result.

- Encoders: AraBERTv2 (text), Wav2Vec2 (audio), Swin-tiny (visual).
- Metrics: macro-F1 headline (imbalance), weighted-F1 for comparability with published
  numbers, confusion matrices.
- Validation: stratified 5-fold, same folds for every model, out-of-fold predictions so the
  strata can be sliced after the fact without leakage.

## Layout

```
src/rp2/data/      corpus adapters — the only place that knows about a specific corpus
       base.py       Sample + DatasetAdapter; congruence is derived here
       armusa.py     Ar-MuSA
       egymer.py     EGY-MER — implemented, data absent, assumptions labelled
       synthetic.py  generated corpus; null control
src/rp2/features.py  frozen encoders + cache
src/rp2/models.py    LateFusion, LateFusionWide, TriCAN
src/rp2/train.py     one CV loop, used by every model
src/rp2/evaluate.py  metrics, stratification, difference-in-differences
experiments/         fetch_armusa.py, extract.py, run.py
tests/               19 tests, incl. two null controls
```
