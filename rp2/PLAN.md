# RP2 — research question and experimental plan

Revised **2026-08-05**, after EGY-MER access failed and a survey of what is actually
obtainable. Supersedes the 2026-08-04 draft, whose three-corpus design assumed EGY-MER
would arrive. See `README.md` for how to run any of this.

---

## 0. What changed, and why

The 2026-08-04 plan was built around EGY-MER as the target corpus, with MORAD as the
fallback. Amany has not responded, and the plan has to survive the possibility that she
never does. So on 2026-08-05 every candidate corpus was checked for **actual
obtainability** rather than publication status — because published is not the same as
downloadable, and that distinction is what sank the original plan.

| Corpus | Task | Modalities | Obtainable? |
|---|---|---|---|
| **Ar-MuSA** | 3-class sentiment, 8,700 | text + audio + **single frame** | ✅ **open**, HF `Skhaled/Ar-MUSA`, AFL-3.0, no gate |
| **MDER-MA** | 4-class emotion, 1,322 | text + audio only — **no visual** | ✅ open, Mendeley, CC BY 4.0 |
| **MORAD** | 7-class emotion, 2,325 | text + audio + video | ⚠️ email the authors, case-by-case |
| **Amd'SaEr** | sentiment + emotion, 1,037 | text + audio + video | ⚠️ repo holds code, not data |
| **EGY-MER** | 7-class emotion, 363 | text + audio + video | ❌ unanswered |

Two consequences:

1. **MDER-MA is not a tri-modal substitute.** Its four "modalities" are audio, text,
   spectrogram and mel-spectrogram — the last two are *derived from the audio*. There is no
   facial or visual stream. It cannot stand in for EGY-MER in a tri-modal study, and the
   earlier note calling it a close alternative was wrong.
2. **MORAD has the same failure mode as EGY-MER** — access is granted case-by-case by email.
   It is worth requesting (one email, longest lead time, do it now) but it cannot be the
   plan's foundation.

**So the study is built on Ar-MuSA, which is open**, and EGY-MER/MORAD become extensions
that slot in if they arrive. The code enforces this: nothing downstream of `rp2.data` knows
which corpus is loaded or how many labels it has.

---

## 1. Sentiment analysis vs emotion recognition

Still the conceptual core, and unchanged.

| | **Sentiment analysis** | **Emotion recognition** |
|---|---|---|
| What it asks | How *positive or negative*? | *Which* affective state? |
| Labels | neg / neu / pos | happiness, sadness, anger, fear, disgust, surprise, neutral |
| Structure | **Ordered** — one axis, valence | **Unordered** — 7 distinct categories |
| Chance accuracy | ~33% | ~14% |
| Example corpus | Ar-MuSA — 8,700 | EGY-MER — 363 · MORAD — 2,325 |

**a. Emotion must discriminate inside what sentiment collapses.** Anger, sadness, fear and
disgust are all "negative". Sentiment analysis is done once it has found that region;
emotion recognition's real work starts there.

**b. The signal lives in different modalities.** Valence is largely *lexical* — "terrible",
"wonderful" carry polarity in the text alone. Discrete emotion is largely *prosodic and
facial*: "I'm fine" said angrily, sadly or fearfully is lexically identical. So the marginal
value of the non-text modalities should be *higher* for emotion than for sentiment.

**c. Imbalance is structural.** Emotion corpora are severely skewed, making macro-F1 the
honest headline metric.

**d. The ceiling is lower.** Absolute scores are not comparable across the two tasks.

---

## 2. The research question

**Dead** — *"no cross-modal attention exists for Arabic."* Falsified by GCMA-Net
(IEEE Access 2026) and two companion papers. See `../lit-review/FINDINGS.md`.

**Also not worth chasing** — *"beat SOTA on Ar-MuSA."* Three 2026 papers already compete
there, best F1 0.8989. A frozen-encoder CPU model would place fourth and prove nothing.

**Alive, and testable today:**

> Cross-modal attention exists to let one modality condition on another. If that is what it
> is doing, its advantage over late fusion should be **concentrated on the samples where the
> modalities carry conflicting evidence**. Nobody has tested this: the three Ar-MuSA papers
> report aggregate F1 only.

**Research question.**

> *Does the advantage of cross-modal attention over late fusion depend on whether the
> modalities agree with each other?*

**And the question RP1 raises, left open pending data:** does that pattern survive the move
from 3-class sentiment on thousands of samples to 7-class emotion on hundreds?

---

## 3. Why Ar-MuSA makes this possible

Ar-MuSA annotates sentiment **separately per modality**, plus overall, plus sarcasm flags for
each. Verified against the released CSVs — all 8,700 rows carry all four sentiment labels,
with no missing values:

| | count | share |
|---|---|---|
| all three modalities agree (**congruent**) | 3,392 | 39.0% |
| exactly two distinct labels (**partial**) | 4,897 | 56.3% |
| all three differ (**conflicting**) | 411 | 4.7% |
| **incongruent** (any disagreement) | **5,308** | **61.0%** |
| flagged sarcastic | 1,157 | 13.3% |
| text disagrees with *both* audio and frame | 1,313 | 15.1% |
| text/audio hard polarity reversal (neg↔pos) | 959 | 11.0% |

Congruence is therefore an **observed property of the corpus**, fixed before any model is
trained. It cannot be tuned to flatter the result. Both strata are large enough for stable
F1 estimates.

Sarcasm and congruence are related but distinct — 66.9% of sarcastic samples are incongruent,
not 100% — so they are reported separately rather than treated as synonyms.

---

## 4. Design

Three models, **identical cached features, identical folds, identical training loop**:

| model | fusion | role |
|---|---|---|
| `late-fusion` | pool per modality → concatenate → classify | the published baselines' strategy |
| `tri-can` | text queries audio and visual, then pool | the proposal |
| `late-fusion-wide` | late fusion widened to Tri-CAN's parameter count | **capacity control** |

The capacity control is the difference between a result and a press release: if Tri-CAN beats
plain late fusion but not the widened one, the gain was parameters, not attention. Measured
match is within 0.25% (1,403,651 vs 1,407,003 params at `d_model=256`).

**Headline statistic — difference-in-differences on macro-F1:**

```
DiD = [F1(TriCAN) − F1(LateFusion)] on incongruent
    − [F1(TriCAN) − F1(LateFusion)] on congruent
```

with a **paired** bootstrap CI: both models are resampled with the same indices, because
they saw the same folds and the same samples. Resampling independently would inflate the
interval with variance the paired design removes.

### Why this design cannot fail

- **DiD > 0** — cross-modal attention helps most where modalities conflict. Supports the
  architectural story, and is a finding nobody has reported on this corpus.
- **DiD ≈ 0** — whatever advantage it has is uniform. That is evidence *against* the
  "attention resolves conflicting evidence" rationale, which is a real and reportable result
  about a widely-assumed mechanism.
- **DiD < 0** — attention helps *least* where modalities conflict, which would be genuinely
  surprising and worth the write-up on its own.

All three are a complete RP2. Stated up front, not as a hedge afterwards.

---

## 5. Protocol

**CPU-only, frozen encoders, features extracted once and cached.** Forced by the hardware,
but also correct: with encoders frozen and features shared, the *only* difference between
baseline and Tri-CAN is the fusion mechanism, so any difference is attributable to it. Lead
with that framing; it is a cleaner ablation than the original LoRA plan, which would confound
fusion changes with encoder adaptation.

- Encoders: AraBERTv2 (text), Wav2Vec2 (audio), Swin-tiny (visual).
- **Metrics:** macro-F1 headline; weighted-F1 for comparability with published numbers;
  confusion matrices. Never compare absolute scores across sentiment and emotion corpora —
  only the *relative* fusion gap within each.
- **Validation:** stratified 5-fold, same folds for every model, out-of-fold predictions so
  strata can be sliced after the fact without leakage.
- **Ablations:** bimodal (T+A vs T+V); attention direction; fusion mechanism.
- **Qualitative:** attention maps on incongruent samples (`return_attention=True`).

### Controls, because a stratified analysis is easy to fool

1. **Capacity-matched baseline** — above.
2. **Null corpus** — the same analysis on synthetic data with no signal must produce chance
   performance and a DiD interval containing zero. Asserted in `tests/test_pipeline.py`; both
   pass.
3. **Out-of-fold predictions** — every sample predicted by a model that never saw it.

---

## 6. Honest caveats

1. **Ar-MuSA's visual modality is a single frame per clip**, not a video — one `.jpg` per
   sample ID. Tri-CAN's visual attention is therefore over Swin's *spatial* patch grid, not a
   temporal sequence. Must not be described as temporal. It is also a weaker visual signal
   than the proposal assumed.
2. **This is sentiment, not emotion.** The task-regime question is left open, explicitly.
3. **Per-modality labels are annotations, not ground truth about the signal.** "The modalities
   disagree" means the annotators assigned different labels per stream — a good proxy for
   conflicting evidence, but a proxy.
4. **The corpus is Movies (5,825) and Food (2,875) reviews**, not general conversation.
5. **8,700 HF rows vs "107 videos / 8,700 samples" in the paper** — samples are utterance
   segments, so multiple rows share a source video. Folds are stratified by label but **not
   grouped by source video**, so some speaker/topic leakage across folds is possible. This
   affects absolute F1; it affects the DiD contrast far less, since it applies equally to
   both models and both strata. Flag it rather than hide it.

---

## 7. Status

- [x] Corpus obtainability survey — the decisive change
- [x] Adapter architecture; EGY-MER adapter written against a documented assumption
- [x] LateFusion / TriCAN / capacity-matched control
- [x] DiD with paired bootstrap
- [x] 19 tests pass, incl. two null controls
- [x] Ar-MuSA congruence statistics verified from the released CSVs
- [ ] Ar-MuSA download (~10 GB, resumable; HF rate-limits, hence the retry loop)
- [ ] Pilot timing → decide whether to subsample
- [ ] Full extraction and the real run
- [ ] Report

**Nour's one action:** email the MORAD authors (Atou et al., `houdaipha` on GitHub) asking
for access — non-commercial academic research, case-by-case. Longest lead time; costs one
email; would let the study extend to 7-class emotion even if EGY-MER never arrives.
