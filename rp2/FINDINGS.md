# RP2 findings

_Corrected 2026-09-26 after re-checking every number against `results/`: EGY-MER provenance now comes from `experiments/label_provenance.py`; anchor deltas, the hybrid repair, and the best-encoder benchmark are stated precisely._

Read this before the meeting. Every number here was produced by code in this
directory; nothing is quoted from memory. Where a claim is weaker than it looks,
that is said in the same sentence.

---

## The story in one page

We set out to test whether cross-modal attention (Tri-CAN) beats late fusion for
Arabic multimodal affect. **It does not** — on two corpora, twelve fusion
arrangements, three anchor choices, four encoder families, frozen and fine-tuned.

Working out *why* produced the actual contribution, in two steps.

**Step 1 — Ar-MuSA cannot test fusion, and we can prove it.** The corpus label
agrees with the corpus's *own transcript annotation* on **96.5%** of samples. A
text-only model already holds nearly all the label information, so no fusion
mechanism has room to distinguish itself. That is a property of the benchmark
and applies to every published result on it, not just ours.

**Step 2 — we predicted what a corpus that CAN test fusion would look like, then
found one.** EGY-MER arrived on 2026-08-09. The same measurement inverts:
transcript **59.4%**, audio **91.1%**, video **84.4%**. The label lives in the
*delivery*. Model behaviour inverts to match, and the multimodal gain triples.
**The prediction held.**

**And yet nothing beats plain concatenation.** On EGY-MER, across twelve
arrangements, no attention design beats late fusion by more than seed noise, and
late fusion is the smallest model (with the base encoders it also scores highest;
with the best encoders a gated hybrid is 0.014 ahead, inside the seed SD). Every gain we
achieved came from **representation** — encoder domain, layer depth, fine-tuning
— and none from fusion design.

**Headline number:** weighted F1 **0.664** on EGY-MER against a published
baseline of **0.519**. That is late fusion, base encoders, seed 0; the 3-seed
mean is 0.654, and 0.669 with the best encoders. (Caveat: their 363 samples,
different splits. Indicative, not like-for-like.)

---

## 1. Label provenance — the measurement everything turns on

How often the overall label equals each modality's *own* annotation:

| | Ar-MuSA (sentiment) | EGY-MER (emotion) |
|---|---|---|
| **transcript** | **96.5%** | 59.4% |
| audio | 77.1% | **91.1%** |
| frame / video | 48.2% | 84.4% |
| overall differs from text label | 3.5% | **40.6%** |

On Ar-MuSA a perfect transcript-label predictor would score macro-F1 **0.9655**
on the "multimodal" target; only 304 of 8,699 clips disagree with the text. The
sarcasm flag is the same (94.9%).

**Say this in the meeting:** *label provenance is cheap to compute and should be
reported whenever a multimodal benchmark is proposed.* It tells you in one number
whether the benchmark can answer the question being asked of it.

## 2. The prediction, confirmed by model behaviour

Unimodal macro-F1 inverts in the same direction as the annotations:

| modality alone | Ar-MuSA | EGY-MER |
|---|---|---|
| text | **0.5458** | 0.3404 |
| audio | 0.4184 | 0.4770 |
| visual | 0.4327 | **0.5225** |
| all three (late fusion) | 0.5593 | 0.5664 |
| **multimodal gain** | **+0.0135** | **+0.0439** |

Three times the multimodal benefit, exactly where the label stops being text.

## 3. Fusion: twelve arrangements, EGY-MER, 3 seeds

| arrangement | macro-F1 | params |
|---|---|---|
| **late-fusion** | **0.5580** ±0.0063 | **891,655** |
| hybrid-gated:text | 0.5566 ±0.0141 | 2,670,601 |
| late-fusion-wide | 0.5525 ±0.0210 | 1,947,463 |
| tri-can:text | 0.5481 ±0.0112 | 1,945,863 |
| hybrid:visual | 0.5453 ±0.0049 | 2,473,223 |
| hybrid:audio | 0.5444 ±0.0067 | 2,473,223 |
| hybrid:text | 0.5411 ±0.0051 | 2,473,223 |
| tri-can:visual | 0.5388 ±0.0167 | 1,945,863 |
| all-pairs (MulT) | 0.5347 ±0.0210 | 6,422,791 |
| gated-fusion | 0.5295 ±0.0097 | 1,089,290 |
| early-fusion | 0.5176 ±0.0099 | 1,681,415 |
| tri-can:audio | 0.5142 ±0.0051 | 1,945,863 |

Largest seed SD is 0.021, so **the top seven are statistically
indistinguishable**. The accurate statement is "no arrangement beats plain
concatenation", not "concatenation is significantly better".

**Same benchmark with the best encoders** (`fusion_benchmark-egy-mer-best.json`):
hybrid-gated:text 0.6081 ±0.0169, late-fusion-wide 0.6022 ±0.0121, late-fusion
0.5935 ±0.0089, ... the three tri-can variants are 10th–12th (0.5587, 0.5535,
0.5444). The top changes order but stays inside seed noise, so the conclusion
holds.

### The design flaw in Tri-CAN

Tri-CAN feeds its classifier `[pool(anchor), pool(anchor's view of A),
pool(anchor's view of B)]` — all three pooled over **anchor positions**. The raw
A and B representations never reach the classifier. It does not add attention to
late fusion; it **replaces** the other modalities with the anchor's
interrogation of them, so it carries strictly less information than the baseline.

That predicts the pattern: on incongruent samples **every** anchor is below late
fusion (text −0.047, audio −0.070, visual −0.066; seed 0). On congruent samples
the text anchor is ahead (**+0.036**). The text-anchor difference-in-differences
is −0.082 but its 95% bootstrap CI [−0.189, +0.034] includes zero: a consistent
direction, not an established effect. On a corpus 56% incongruent, the losses
dominate.

**The repair works, and only to parity.** `hybrid-gated` restores the discarded
representations (input is a strict superset of late fusion's), lifting
text-anchored attention from −0.0099 to **−0.0014** — from losing to tied. The
diagnosis is confirmed; the attended view simply adds nothing the raw vectors
did not already carry.

## 4. Where the gains actually are

| change | gain | corpus |
|---|---|---|
| dialect-matched text encoder (AraBERT-twitter over AraBERTv2) | **+0.054** | Ar-MuSA |
| audio read mid-stack instead of final layer | **+0.053** | EGY-MER |
| fine-tuning the text encoder | **+0.051** | Ar-MuSA |
| expression-pretrained visual encoder over ImageNet Swin | **+0.026** | EGY-MER |
| **best of twelve fusion architectures** | **≈ 0** | both |

Fine-tuning improved late fusion and Tri-CAN by **exactly +0.0505 each**, leaving
their order unchanged — so the fusion null is not an artefact of frozen features.

## 5. Four hypotheses we predicted and had to reject

Each was stated before running. Reporting them is part of the method.

1. **Anchor choice.** Text is EGY-MER's weakest modality, so anchoring on audio
   or vision should help. It did not — text stayed the best anchor
   (audio −0.034, vision −0.009 relative to the text anchor).
2. **Multilingual audio encoder.** XLS-R-300m scored *worse* than English
   wav2vec2-base (0.3835 vs 0.4206). The weak audio was not a language mismatch.
3. **Face cropping.** Cost **−0.0375** despite 98% detection. EGY-MER is
   speaker-*diarised* but not speaker-*localised*: the largest face in a
   talk-show frame is often not the person speaking, so cropping reliably
   delivers the wrong face. Active-speaker detection would be needed.
4. **Emotion-pretrained audio encoder.** SUPERB-ER scored worse than the generic
   model (0.4578 vs 0.4770) — it is tuned on English *acted* emotion, and that
   specialisation narrows the representation.

**The lesson 2 and 4 share:** matching an encoder's *domain* to the data helps;
matching its *task label space* does not.

## 6. Controls that make the nulls trustworthy

- **Capacity-matched baseline** — late fusion widened to Tri-CAN's parameter
  count, so "it was just bigger" is answerable.
- **Null corpus** — the same analysis on signal-free synthetic data yields chance
  performance and a DiD interval containing zero (2 tests).
- **Out-of-fold predictions** — every sample predicted by a model that never
  trained on it.
- **Repeated seeds** on EGY-MER, with ties declared rather than hidden.
- **Identical folds, features and training loop** across every model compared.

## 7. Limitations — state these before being asked

1. **484 samples, 7 classes**, Fear at 12 and Happiness at 18. Macro-F1 is noisy;
   differences below the seed spread are not interpreted.
2. **The 0.519 comparison is indicative**, not like-for-like — different sample
   count and splits.
3. **Only the text encoder was fine-tuned** (compute limit, not a design choice).
4. **Per-modality labels are annotations, not ground truth about the signal** —
   a good proxy for conflicting evidence, but a proxy.
5. **Folds are not grouped by source video**, so some speaker leakage is
   possible. Affects absolute scores more than within-study contrasts.
6. **EGY-MER as delivered is not the published corpus**: 5,471 segments with
   media, 505 annotated, against the paper's 363. Worth asking the holders about.

## 8. Anticipated questions

**"Your architecture didn't work — isn't that a failed project?"**
The architecture is one finding of several. We showed the standard Arabic
benchmark cannot evaluate fusion at all, predicted what a corpus that could would
look like, confirmed it on EGY-MER, and beat the published baseline by 0.145
weighted F1. We also found and fixed a concrete design flaw in the proposed
architecture. A negative architectural result with an explanation is a result.

**"Why should we believe the null?"**
Twelve arrangements, three seeds, a capacity-matched control, and a null-corpus
test asserting the analysis invents no effect from noise. And the null survives
fine-tuning, which is the usual escape hatch.

**"Who suggested cross-modal attention?"**
The published literature: MulT (Tsai et al. 2019) established it for multimodal
sentiment and is one of our four seed papers; an independent group published
GCMA-Net applying it to Arabic in 2026. We tested a widely-adopted assumption and
found its limit.

**"What would you do next?"**
Active-speaker detection so visual cropping becomes usable; a grouped-by-video
split; and label provenance reported for MORAD, which has the same 7-emotion
taxonomy and 6× the data.
