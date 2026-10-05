# RP2 — code, results and documents

The report is [`report/main.pdf`](report/main.pdf); the slides are
[`slides/presentation.pdf`](slides/presentation.pdf) (speaker notes in
`slides/presentation-notes.pdf`). This file explains how the code behind them is organised
and which script produced which number.

## What was done, in short

RP1 proposed Tri-CAN, a text-anchored cross-modal attention network, and predicted it would
beat late fusion on Egyptian Arabic. RP2 tested that on two corpora:

- **Ar-MuSA** (8,699 clips, 3-class sentiment, open download). The overall label matches the
  transcript annotation on 96.5% of clips, so there is almost nothing for fusion to add.
- **EGY-MER** (484 labelled segments, 7-class emotion, received on 9 August 2026). Here the
  label lives mostly in audio and video, so fusion does matter.

On neither corpus did any of twelve fusion arrangements beat plain concatenation by more than
the seed spread. The measurable gains all came from the encoders (dialect-matched text model,
mid-layer audio features, expression-pretrained visual model, fine-tuning). `FINDINGS.md`
has the full story with every number; `PLAN.md` is the experimental plan as it stood on
5 August, before EGY-MER arrived, kept for the record.

## Protocol

CPU only. Encoders are frozen; features are extracted once and cached, and only the fusion
module and classifier are trained. With identical features, folds and training loop, the
fusion mechanism is the only thing that differs between the models being compared.
Stratified 5-fold cross-validation, out-of-fold predictions, macro-F1 as the main metric,
EGY-MER results averaged over 3 seeds.

## Where each result comes from

| Report / slides | Script | Output in `results/` |
|---|---|---|
| Unimodal and late-fusion scores (Tables 2, slides 5 and 8) | `experiments/run.py` | `ar-musa-seed0.json`, `egy-mer-seed0.json`, `unimodal-*/`, `egymer-*uni-*/` |
| Label provenance (Table 1, slides 6–7) | `experiments/label_provenance.py` | `label_provenance.json` |
| Twelve fusion arrangements (Section 4, slide 9) | `experiments/fusion_benchmark.py` | `fusion_benchmark-egy-mer.json`, `fusion_benchmark-egy-mer-best.json` |
| Anchor choice and congruent/incongruent split (slides 10, 12, 17) | `experiments/anchor_ablation.py` | `anchor_ablation.json` |
| Audio layer depth (Table 3) | `experiments/layer_sweep.py` | `layer_sweep-*.json` |
| Encoder comparison (Table 3, Section 6) | `experiments/encoder_sweep.py`, `audio_encoder_check.py` | printed; feature caches |
| Frozen vs fine-tuned text encoder (Table 3) | `experiments/frozen_vs_finetuned.py` | `frozen_vs_finetuned-*.json` |
| Face cropping (Section 6) | `experiments/face_crop_test.py` | `face_crop_test.json` |
| Predicting incongruence on Ar-MuSA | `experiments/incongruence_study.py` | `incongruence.json` |
| Synthetic null control | `experiments/run.py --dataset synthetic`, `tests/` | `synthetic-seed0.json` |

Each script's docstring gives its exact command line and the reason it was run.

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio torchvision
uv pip install --python .venv/bin/python \
    transformers scikit-learn pandas numpy pillow soundfile librosa huggingface_hub pytest \
    openpyxl opencv-python-headless audioread

# pipeline check, no data needed (~1 minute)
.venv/bin/python -m experiments.run --dataset synthetic
.venv/bin/python -m pytest tests            # 19 tests, incl. two null controls (-m slow runs only those)

# Ar-MuSA: download (~10 GB, resumable), extract features, run
.venv/bin/python -m experiments.fetch_armusa --root data/raw/ar-musa
.venv/bin/python -m experiments.extract --dataset ar-musa --root data/raw/ar-musa --out data/cache/ar-musa
.venv/bin/python -m experiments.run --dataset ar-musa --cache data/cache/ar-musa

# EGY-MER: same, with the corpus placed under data/raw/egy-mer
.venv/bin/python -m experiments.extract --dataset egy-mer --root data/raw/egy-mer --out data/cache/egy-mer
.venv/bin/python -m experiments.run --dataset egy-mer --cache data/cache/egy-mer
```

The corpora and feature caches are not in this repository. Ar-MuSA is public on Hugging Face
(`Skhaled/Ar-MUSA`). EGY-MER is distributed by its authors on request, so it cannot be
redistributed here; `src/rp2/data/egymer.py` documents the layout of the delivered copy.

## Layout

```
src/rp2/data/        corpus adapters (Ar-MuSA, EGY-MER, synthetic)
src/rp2/features.py  frozen encoders and feature cache
src/rp2/models.py    late fusion, widened late fusion, Tri-CAN, and the other arrangements
src/rp2/train.py     the cross-validation loop shared by every model
src/rp2/evaluate.py  metrics, congruence strata, difference-in-differences with bootstrap CI
src/rp2/finetune.py  text-encoder fine-tuning
experiments/         one script per experiment (table above)
results/             raw outputs behind every reported number
data/models/         YuNet face detector (OpenCV model zoo) used for face cropping
report/, slides/     LaTeX sources and PDFs
```
