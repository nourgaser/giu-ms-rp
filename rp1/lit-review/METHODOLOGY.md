# How this literature search was done

Written to be shown to Dr. Nada, and to brief Nour before that conversation.

The claim being made is **not** "we read 1,322 papers." It is:

> *A scripted search across three bibliographic databases, plus citation snowballing,
> retrieved 1,322 unique records. Those were triaged by transparent rules into priority
> tiers. The 111-paper core tier was screened at title and abstract with a written reason
> per decision, giving 68 included papers.*

That is standard PRISMA vocabulary. In PRISMA, **"identified" ≠ "read."** Systematic reviews
routinely identify thousands of records and include a few dozen — the large number describes
the *net cast*, and the small number describes the *catch*. Conflating them is the mistake;
keeping them separate is the whole point of reporting both.

---

## What was actually done, in order

**1. A written protocol, before searching** (`protocol.md`)
Research questions, sources, search blocks, inclusion/exclusion criteria, and coverage window
(inception → 2026-07-31) were fixed in advance. Writing the protocol first is what stops the
criteria from being retrofitted to whatever the search happened to return.

**2. Keyword search across three independent bibliographic databases**

| Database | Why it's there |
|---|---|
| **OpenAlex** | Primary. ~250M works; absorbed Microsoft Academic Graph; full citation graph; open |
| **arXiv** | Preprints. The only source current enough to see Q1/Q2 2026 |
| **Crossref** | Authoritative DOI, venue, and date metadata; independent cross-check |

Three sources rather than one because each has a known blind spot: OpenAlex under-indexes
very recent preprints, arXiv has no peer-reviewed venue coverage, Crossref carries no
abstracts. Where they disagree, the disagreement is recorded, not silently resolved.
(Semantic Scholar was attempted and returned HTTP 429 without an API key — recorded as a
gap rather than quietly dropped.)

20 queries across 5 thematic blocks. **1,596 raw records → 1,147 after deduplication.**

**3. Citation snowballing — the step that matters most**
Keyword search only finds papers that use your vocabulary. Citation traversal finds the ones
that don't. From four confirmed seed papers (EGY-MER, Ar-MuSA, UniTextFusion, MulT), the
citation graph was walked in both directions: everything they cite, and everything citing them.

**This surfaced 176 papers the keyword search missed entirely.** That number is the direct,
quantified answer to "how do you know you didn't miss something?"

Merged and deduplicated: **1,322 unique records.**

**4. Rule-based triage into priority tiers — not screening**
Each record was scored by keyword rules over title + abstract into P1–P6 (P1 = Arabic +
affect + multimodal; P6 = peripheral). **The rules are published, and every row records the
exact terms that produced its tier**, so any assignment can be checked or disputed.

This is explicitly *not* a screening decision — a tier is a reading order, assigned by
keyword rules that cannot judge relevance. Screening is a separate step, and the `Decision`
column is filled there, never by the triage rules.

**5. Title/abstract screening**
All 111 P1 papers were screened against the predefined criteria, with a **recorded reason for
every decision** (`screening_decisions.py`, applied to the workbook by `apply_screening.py`).
The reasons are in the sheet, so any individual call can be inspected and disputed rather
than taken on trust.

Outcome: 111 screened → **68 included** (26 core Arabic, 20 method-lineage, 22 marginal),
43 excluded with reasons. Full-text assessment is still outstanding, so the included set is
provisional.

`apply_screening.py` refuses to write unless every paper receives exactly one decision and
every decision matches exactly one paper — a partial match would silently corrupt the counts.

**6. Bibliography built from publishers, not by hand**
Every included paper with a DOI had its BibTeX fetched from the publisher of record by DOI
content negotiation at `doi.org` (`make_bib.py`). Authors, venue, volume and year are
therefore correct by construction. Two included works are theses with no DOI; their metadata
was verified by hand against the issuing institutions' repositories (AUS and KFUPM) and
recorded with a note stating what was and was not confirmed.

---

## Tools

| Tool | Used for |
|---|---|
| OpenAlex / Crossref / arXiv APIs | record retrieval, metadata, citation graph |
| Python (`httpx`, `pandas`, `rapidfuzz`, `openpyxl`) | search, dedup, triage, spreadsheet |
| DOI content negotiation (`doi.org`) | publisher-authoritative BibTeX |
| Claude Code | writing the scripts, designing the query blocks, first-pass screening |
| Tectonic | LaTeX build |

Claude Code was used throughout, and openly: it wrote the retrieval and triage code, helped
design the search blocks, and produced the first pass over the 111 core titles and abstracts,
which was then reviewed. AI assistance is permitted on this course; the point of saying so is
that the method should be judged on what it produces, and everything it produced is on disk.

**The integrity rule, stated in `protocol.md` and enforced by the tooling:** every
bibliographic fact — title, year, venue, authors, DOI — comes from an API record cached on
disk or from the publisher of record, never from a language model's recollection. Hallucinated
citations are the specific failure mode that makes AI-assisted reviews untrustworthy, and the
pipeline is built so that a citation cannot enter the bibliography without a retrieved record
behind it.

---

## Why this is more trustworthy than a manual search, not less

1. **It is reproducible.** `python search.py --refresh` re-runs the whole thing. A manual
   afternoon on Google Scholar cannot be re-run by anyone, including the person who did it.
2. **It is auditable.** Every API response is cached verbatim in `data/raw/` (25 MB, 100+
   files). "Where did this paper come from?" is answered by a file, not a memory.
3. **The search strategy is written down.** Exact queries, date windows, and sources are in
   `protocol.md` — the reader can judge the coverage, and can find the holes.
4. **It states its own gaps.** Semantic Scholar missing, OpenAlex citation counts running
   lower than Google Scholar's, some records lacking abstracts — all recorded.
5. **Retrieval and judgement are separated.** Retrieval and triage are mechanical and
   rule-based; screening decisions carry a written reason each and are open to challenge.

## Honest limitations — state these before being asked

- **1,322 records were retrieved, not read.** The number describes coverage, nothing more.
- **Triage is keyword rules over title + abstract.** Rules are crude: a paper whose abstract
  omits the word "multimodal" can land in the wrong tier. That is why the tiers are a reading
  *order*, not a verdict, and why P2/P3 still get sampled rather than discarded unseen.
- **Screening was at title and abstract only.** Full texts have not been assessed, so the
  68 included are provisional.
- **Semantic Scholar is absent** — rate-limited without a key. Its unique coverage is a
  known, declared hole.
- **Citation counts are OpenAlex's** and run systematically lower than Google Scholar's. Used
  for ranking within the pool, never quoted as a paper's true citation count.
- **"EGY-MER has no citations"** means *no citing work indexed in OpenAlex as of 2026-08-04,
  and no mention across our 1,322-record corpus.* It is strong evidence, not proof.
- **Coverage ends 2026-07-31.** Anything published in August 2026 is outside the window.
- **Two included works are DOI-less theses**, verified by hand rather than by API. Both are
  superseded by peer-reviewed articles that are also in the included set.

---

## How to explain this in the meeting

**The one-liner:** *"We used the same search infrastructure systematic reviews use —
OpenAlex, Crossref, arXiv, plus citation snowballing — scripted so it's reproducible. It
retrieved 1,322 unique records. We triaged those by published rules and screened the top tier
with a written reason per paper. The scripts and every raw API response are in the repo."*

Anticipated questions:

**"Did you actually read all of these?"**
No — and the review doesn't claim to. 1,322 is what the search *identified*. We screened the
111-paper core tier and included 68. That identified/included split is standard PRISMA
reporting.

**"How do you know you didn't miss the important work?"**
Two ways. Three independent databases rather than one. And citation snowballing from the four
seed papers — forward and backward — which found 176 papers keyword search missed. That's
also how we caught GCMA-Net, the 2026 paper that forced us to narrow our own novelty claim.

**"Why is your novelty claim different from the RP1 draft?"**
Because the search found work published after the draft was written. GCMA-Net (IEEE Access,
2026) applies gated cross-modal attention to Arabic multimodal sentiment. We narrowed the
claim to what is still true: that work is on 3-class *sentiment* with 8,700 samples; nothing
has been done on 7-class *emotion* on EGY-MER, which is cited by nothing. Finding that
ourselves — and correcting the claim — is the search doing its job.

**"How much of this was AI?"**
The scripts were written with Claude Code, and it did the first screening pass over the 111
core papers with a written reason for each. Everything it produced is inspectable in the
sheet and in the repo. No citation comes from a model — every bibliographic record was
fetched from a database or the publisher, and cached. The tool sped the work up; it isn't
what the work rests on.

**"Can I see the sheet?"**
`RP1_paper_tracking.xlsx` — Method, PRISMA counts, Screening (all 1,322 with tiers, the
terms that assigned them, and a decision + reason for the 111 core papers), Extraction
(68 included, pre-seeded), Sources.

---

## Reproducing this

```bash
cd rp1/lit-review
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python httpx pandas rapidfuzz openpyxl
.venv/bin/python search.py       # 3 databases x 20 queries -> data/pool.csv
.venv/bin/python snowball.py     # citation graph from 4 seeds -> data/snowball.csv
.venv/bin/python make_sheet.py   # merge, triage -> RP1_paper_tracking.xlsx
.venv/bin/python make_bib.py     # publisher BibTeX for the included set
```

Cached responses in `data/raw/` make re-runs offline and identical. `--refresh` re-queries.
