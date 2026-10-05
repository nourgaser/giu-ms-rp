# Search findings — 2026-08-04

Corpus: **1,322 unique papers** (1,596 raw keyword hits → 1,147 deduped, plus 176 found
only by citation snowballing). Coverage to 2026-07-31. Current `references.bib`: 11 entries.

Priority tiers: P1 core (Arabic + affect + multimodal) **111**, of which **46 from 2025 and
23 from 2026**. P2 229 · P3 265 · P4 70 · P5 218 · P6 429.

---

## 🔴 RP1's Gap #4 is falsified as written

`main.tex` § Gap Analysis, item 4 claims:

> *"Absence of Cross-Modal Attention for Arabic … no equivalent investigation exists for
> Arabic, let alone for dialectal Egyptian Arabic."*

That is no longer true. Verified, with DOIs:

| Paper | Venue | Dataset | Result |
|---|---|---|---|
| **GCMA-Net: A Gated Cross-Modal Attention Network for Arabic Multimodal Sentiment Analysis** — Ben Cheikhi & Nfaoui | **IEEE Access 2026** · `10.1109/access.2026.3667314` | Ar-MuSA | Acc 80.17%, **F1 0.8015** |
| A Directional Semantic Enhancement Approach with Gated Fusion for Multimodal Arabic SA — Ben Cheikhi, Nfaoui, Elbiach | MAKE 2026 · `10.3390/make8050139` | Ar-MuSA | Acc 89.89%, **F1 0.8989** |
| A Transformer-Based Approach for Multimodal Arabic Sentiment Analysis — Ben Cheikhi & Nfaoui | IJACSA 2026 · `10.14569/ijacsa.2026.0170290` | Ar-MuSA | **F1 0.7756** (early fusion, ViT+MarBERT+HuBERT) |
| Alignment Challenges in Arabic Multimodal Sentiment Analysis — Boukhalda, Chouikhi, Jarray | ICAART 2026 · `10.5220/0014250400004052` | — | (no abstract indexed) |

GCMA-Net is close to Tri-CAN's stated contribution: "modality-specific projection,
**cross-modal attention**, and **gated fusion**" over text/audio/visual for Arabic.

**This must be fixed before the meeting.** A supervisor who searches for five minutes finds
GCMA-Net, and an unqualified novelty claim is precisely the kind of thing that turns a
closing meeting into another round.

## 🟢 But a narrower gap survives, and it is defensible

Everything above targets **Ar-MuSA** — 8,700 samples, **3-class sentiment** (pos/neg/neutral),
Moroccan/MSA. None of it touches **EGY-MER** — 363 samples, **7-class emotion**, Egyptian
dialect, severe imbalance (Fear/Surprise <4%).

**EGY-MER is cited by nothing.** Of 1,322 papers, exactly one mentions it: itself.

So the honest restatement is:

> Cross-modal attention has now reached Arabic multimodal *sentiment* (3-class, Ar-MuSA),
> but not Arabic multimodal *emotion recognition* — a harder task with a finer label space,
> an order of magnitude less data, and severe class imbalance. Whether the gains reported on
> Ar-MuSA survive that regime is untested.

That is narrower than the original claim but it is **true**, it is **verifiable**, and it is
still enough to carry an RP2. It also makes the write-up stronger: RP1 now demonstrates
command of literature published *while the project was dormant*, which reframes the delay.

## Strategic consequence: do not fall back to Ar-MuSA

This morning Ar-MuSA looked like the natural fallback if EGY-MER never arrives. It now looks
like the worst option: three 2026 papers already compete on it and SOTA is **F1 0.8989**.
Entering there means placing fourth on a crowded benchmark.

Revised fallback ranking:

1. **EGY-MER** — baseline F1 0.519, zero competitors. Still first choice. *Blocked on Amany.*
2. **MORAD** (`10.1109/tcss.2026.3688593`, IEEE TCSS 2026) — **strong new option.**
   2,325 videos, 3 hours, 329 speakers, unscripted, Moroccan Darija, and **the same seven
   emotions as EGY-MER**. Six times larger than EGY-MER and brand new, so also uncontested.
   The research question transfers almost unchanged — only the dialect changes.
3. MSA-Moroccan Dialect (`10.17632/kfjztyzztb`, Mendeley) — 3,040 podcast samples, but
   sentiment labels, so the question would need restating.
4. Ar-MuSA — **deprioritised**, for the reason above.

Other 2026 Arabic multimodal resources logged for completeness: Arabic Multimodal Dataset for
Aspect-Based SA (IEEE DataPort), AraImgStory-10K (Zenodo).

## What to do with this

1. Rewrite Gap #4 in `main.tex` to the narrower claim, and **cite the four papers above** in
   Related Work. Turning the threat into a citation is what makes it a strength.
2. Add a `Recent Advances (2026)` subsection — the review currently stops at 2025.
3. Reprice the Tri-CAN contribution: the novelty is the *task regime* (7-class dialectal
   emotion, tiny imbalanced data), not the mechanism.
4. Chase EGY-MER; keep MORAD as a live, pre-vetted alternative rather than a panic option.

## Caveats

- Priority tiers are keyword-rule triage, not screening decisions. (The P1 tier has since
  been screened — see the screening section below. P2-P6 have not been.)
- Citation counts come from OpenAlex and run lower than Google Scholar's.
- Semantic Scholar was unreachable (HTTP 429 without a key), so its unique coverage is absent.
  OpenAlex + Crossref + arXiv + snowballing is good coverage but not exhaustive.
- "Alignment Challenges in Arabic MSA" has no indexed abstract — needs manual retrieval.

---

# Screening findings — 111 P1 papers, 2026-08-04

68 included (26 core Arabic, 20 method-lineage, 22 marginal), 43 excluded with reasons.
First pass by Claude, for Nour to adjudicate — see `METHODOLOGY.md` § step 5.

## 🟢 The EGY-MER blocker just became much less dangerous

Screening surfaced **seven Arabic multimodal affective resources absent from the current
bibliography**. This materially de-risks RP2: EGY-MER is no longer a single point of failure.

| Resource | Year | Venue | What it is |
|---|---|---|---|
| **Amd'SaEr** | 2025 | **ACM TALLIP** | Arabic multimodal dataset for **both** sentiment *and* emotion. Strongest venue of the set. |
| **MDER-MA** | 2025 | Data in Brief | Multimodal emotion, low-resource **Moroccan** Arabic |
| **MORAD** | 2026 | IEEE TCSS | 2,325 videos, 329 speakers, Moroccan, **same 7 emotions as EGY-MER** |
| **Multi-Modal Spoken-Arabic** | 2025 | — | Arabic multimodal emotion, dialect-aware |
| **AraSER-Fusion** | 2025 | Arab. J. Sci. Eng. | Arabic multimodal SER, audio–text joint embeddings |
| **AraMABSA** | 2026 | IEEE DataPort | 1,721 Arabic text–image pairs, aspect-based |
| **EYASE** (via a 2026 SER paper) | — | — | **Egyptian** Arabic speech emotion, 4 classes |

**Amd'SaEr and MDER-MA are the priority follow-ups** — both are emotion-labelled Arabic
multimodal corpora in citable venues. Verify obtainability for each, exactly as for EGY-MER.

## Prior work on Egyptian Arabic specifically

**ATFusion** (2023) — audio+text multimodal sentiment for English *and Egyptian Arabic*
dialects. The closest prior work to our target dialect and modality set outside EGY-MER
itself, and not cited in the draft.

## Three papers that constrain how we describe Tri-CAN

1. **Cross-Modality Gated Attention Fusion** (2022) — the direct methodological ancestor of
   GCMA-Net. Useful: it shows the 2026 Arabic work *applies an established mechanism* rather
   than inventing one, which is a fair and defensible way to frame our own contribution.
2. **Text-oriented Modality Reinforcement Network** (2023) and **Hierarchical Text-Guided
   Refinement Network** (2025) — both use text as the anchoring modality guiding the others.
   That is Tri-CAN's premise. **Both must be cited and explicitly distinguished**, or a
   reviewer will ask what is different.
3. **Tri-CLT** (2024) — tri-modal contrastive + transformer. Name collision close enough to
   Tri-CAN to invite confusion; distinguish, or consider renaming.

Also: **Multimodal Arabic Sarcasm Detection using CNN and BiLSTM** (2025, on Ar-MuSA) is the
only multimodal Arabic sarcasm work found, and the draft's RQ4 section currently rests on a
single text-only source. And **"Arabic Multimodal Machine Learning: Datasets, Applications,
Approaches, and Challenges"** (arXiv 2025) is a dedicated survey of exactly our area — the
natural anchor for the landscape section, and a check on our own coverage.

## Still outstanding

- Full-text assessment: the 68 are provisional, screened on abstracts only.
- Three included papers have **no indexed abstract** and need manual retrieval: *Alignment
  Challenges in Arabic MSA*, *Multimodal Arabic emotion recognition using deep learning*
  (Speech Communication), *AraSER-Fusion*.
- `MulMoSenT` and `CLARA` each appear as both a paper and a data/software record — cite once.
