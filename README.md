# Cross-Modal Attention for Egyptian Arabic Multimodal Emotion Recognition

Research Project I and II — German International University, Department of Computer Science.

**Nour Gaser, Mohamed Wael**

## Documents

| | Report | Slides |
|---|---|---|
| **RP1** — systematic literature review and research proposal (Winter 2025/26) | [`rp1/report/main.pdf`](rp1/report/main.pdf) | [`rp1/slides/presentation.pdf`](rp1/slides/presentation.pdf) |
| **RP2** — experiment and results (Spring 2026) | [`rp2/report/main.pdf`](rp2/report/main.pdf) | [`rp2/slides/presentation.pdf`](rp2/slides/presentation.pdf) |

## Summary

**RP1** reviews Arabic multimodal sentiment and emotion recognition through July 2026
(1,322 records from OpenAlex, arXiv and Crossref plus citation snowballing; 111 screened;
68 included) and proposes Tri-CAN, a cross-modal attention network in which text features
query audio and video features, as an alternative to the late-fusion baseline on EGY-MER.

**RP2** tests that proposal on two Arabic corpora, Ar-MuSA and EGY-MER. Cross-modal attention
did not beat late fusion on either. The report explains why: on Ar-MuSA the overall label is
almost entirely recoverable from the transcript annotation (96.5%), so no fusion method can
show an advantage there, and on EGY-MER, where fusion does help, plain concatenation was as
good as every attention design tried. The improvements we did measure came from encoder
choices.

## Repository layout

```
rp1/
  report/       LaTeX source and PDF of the RP1 report; references.bib is shared with RP2
  slides/       RP1 presentation
  lit-review/   the systematic search: scripts, cached API responses, screening decisions,
                and the paper-tracking spreadsheet (RP1_paper_tracking.xlsx).
                METHODOLOGY.md describes the protocol and how to re-run it.
rp2/
  report/       LaTeX source and PDF of the RP2 report
  slides/       RP2 presentation, with a speaker-notes version
  src/, experiments/, tests/   the experiment code
  results/      raw outputs behind every number in the RP2 report and slides
  README.md     how the code is organised and which script produced which result
  FINDINGS.md   the full results narrative
```

## Building the documents

All documents compile with [Tectonic](https://tectonic-typesetting.github.io/) (0.15):

```bash
cd rp1/report && tectonic -X compile main.tex
cd rp1/slides && tectonic -X compile presentation.tex
cd rp2/report && tectonic -X compile main.tex
cd rp2/slides && tectonic -X compile presentation.tex
```

## Data

Neither corpus is included. Ar-MuSA is publicly available on Hugging Face
(`Skhaled/Ar-MUSA`). EGY-MER (Hussein et al., 2025) is distributed by its authors on request.
See `rp2/README.md` for how the code expects each corpus to be laid out.
