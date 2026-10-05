# RP1 Literature Review — Search Protocol

The RP1 draft's own closing note concedes the review "is not systematic and short."
This protocol fixes that: a documented, reproducible, auditable search, so the review
section can honestly be *called* systematic and the coverage claim can be defended.

Registered: 2026-08-04. Coverage target: inception → **2026-07-31** (Q1/Q2 2026 inclusive).

## Research questions

- **RQ1** — What multimodal datasets exist for Arabic, and specifically for dialectal /
  Egyptian Arabic, sentiment and emotion recognition?
- **RQ2** — Which *fusion strategies* have been applied to Arabic multimodal sentiment /
  emotion, and does cross-modal attention appear among them?
- **RQ3** — What is the state of cross-modal attention in high-resource languages, and what
  evidence exists on transferring it to morphologically rich / low-resource languages?
- **RQ4** — How is sarcasm / modality incongruence handled in Arabic affective computing?

## Sources

| Source | Role | Access |
|---|---|---|
| **OpenAlex** | Primary. ~250M works (absorbed Microsoft Academic Graph), full citation graph | free, no key, polite pool via `mailto` |
| **arXiv** | Preprints — the only source current enough for Q1/Q2 2026 | free, no key |
| **Crossref** | Cross-check + DOI/venue metadata completion | free, no key |
| **Semantic Scholar** | Best-effort supplement | aggressively rate-limited without a key; treated as optional |

Using several sources is deliberate: OpenAlex under-indexes very recent preprints, arXiv has
no peer-reviewed venue coverage, and Crossref has no abstracts. Together they close each
other's gaps. Disagreements between them are recorded rather than silently resolved.

## Search blocks

Keyword search alone reliably misses work that uses different vocabulary, so it is paired
with citation snowballing (below).

- **A — Arabic multimodal affect (core):** arabic × multimodal × (sentiment | emotion)
- **B — Dialectal:** (egyptian | dialect*) arabic × (emotion | sentiment | speech)
- **C — Fusion methods:** cross-modal attention, multimodal transformer, modality fusion
- **D — Resources:** arabic multimodal (dataset | benchmark | corpus)
- **E — Adjacent:** arabic (sarcasm | irony) multimodal; arabic speech emotion recognition;
  arabic vision-language / LMM

## Snowballing

From confirmed seeds — EGY-MER, Ar-MuSA, UniTextFusion, AMMD, MulT (Tsai et al. 2019):

- **Backward** — everything each seed cites.
- **Forward** — everything citing each seed. This is what surfaces the 2026 work that
  keyword queries miss, and it is the strongest defence against "you missed X."

## Screening

**Include** if the work is peer-reviewed or a citable preprint, is in English or Arabic,
and addresses (a) Arabic multimodal affect, (b) an Arabic multimodal resource, (c) a fusion
mechanism relevant to RQ2/RQ3, or (d) Arabic sarcasm/incongruence.

**Exclude** — text-only Arabic sentiment with no other modality (unless it defines a
baseline we cite); non-affective multimodal Arabic (OCR, captioning, hate-meme detection)
unless method-relevant to RQ3; duplicates; works with no retrievable abstract *and* no
retrievable full text; venue-less non-arXiv grey literature.

Every exclusion is recorded **with its reason** in the tracking sheet — that record is what
makes the PRISMA counts reproducible.

## Outputs

- `data/raw/` — verbatim API responses, one file per query × source. Never edited.
- `data/pool.csv` — deduplicated candidate pool (DOI match, then fuzzy title ≥ 92).
- `RP1_paper_tracking.xlsx` — the screening + extraction sheet.
- PRISMA counts (identified → deduplicated → screened → eligible → included) for `main.tex`.

## Integrity rules

- Nothing enters `references.bib` that was not returned by a logged query or a snowball
  step, with a real DOI/arXiv ID. No citation is ever written from memory.
- Metadata (year, venue, authors) comes from the API record, not from recollection.
- Where sources disagree on a field, prefer Crossref/OpenAlex DOI metadata and note it.
