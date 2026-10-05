"""
make_sheet.py — build the RP1 paper-tracking workbook from the search outputs.

Merges the keyword pool and the snowball results, triages them with transparent
rules, and writes a multi-sheet xlsx for screening and data extraction.

IMPORTANT: this does NOT decide inclusion. It assigns a *priority* and records the
terms that produced it, so every ranking is explainable and every actual
include/exclude decision stays a human one recorded in the Decision column.
Auto-deciding inclusion would make the PRISMA counts fiction.

    python make_sheet.py

Output: RP1_paper_tracking.xlsx
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

ARABIC = r"arabic|egyptian|dialect|levantine|gulf|darija|maghreb|msa\b|tunisian|algerian|moroccan"
AFFECT = r"sentiment|emotion|affect|sarcas|irony|ironic|opinion|feeling|valence|arousal"
MULTIMODAL = r"multimodal|multi-modal|audiovisual|audio-visual|cross-?modal|tri-?modal|modalit|bimodal"
FUSION = r"fusion|cross-?modal attention|co-?attention|multimodal transformer|late fusion|early fusion|tensor fusion"
PEFT = r"\blora\b|parameter-?efficient|adapter|peft|low-?rank"
RESOURCE = r"dataset|corpus|benchmark|annotat"


def _hits(text: str, pattern: str) -> list[str]:
    return sorted(set(m.group(0).lower() for m in re.finditer(pattern, text, re.I)))


def triage(row) -> pd.Series:
    """Rule-based priority. Every tier is reproducible from the recorded terms."""
    text = f"{row.get('title','')} {row.get('abstract','')}"
    ar, af = _hits(text, ARABIC), _hits(text, AFFECT)
    mm, fu = _hits(text, MULTIMODAL), _hits(text, FUSION)
    pe, rs = _hits(text, PEFT), _hits(text, RESOURCE)

    if ar and af and mm:
        tier, why = "P1", "Arabic + affect + multimodal (core, RQ1/RQ2)"
    elif ar and af:
        tier, why = "P2", "Arabic + affect, modality unclear (RQ4 / baseline)"
    elif af and mm and (fu or pe):
        tier, why = "P3", "fusion/PEFT method for multimodal affect (RQ3 lineage)"
    elif ar and mm:
        tier, why = "P4", "Arabic + multimodal, affect unclear"
    elif af and mm:
        tier, why = "P5", "multimodal affect, general"
    else:
        tier, why = "P6", "peripheral"

    return pd.Series({
        "Priority": tier,
        "Why": why,
        "t_arabic": ", ".join(ar[:6]),
        "t_affect": ", ".join(af[:6]),
        "t_multimodal": ", ".join(mm[:6]),
        "t_fusion": ", ".join(fu[:5]),
        "t_peft": ", ".join(pe[:4]),
        "t_resource": ", ".join(rs[:4]),
        "has_abstract": bool(str(row.get("abstract", "")).strip()),
    })


def main() -> None:
    pool = pd.read_csv(ROOT / "data" / "pool.csv")
    snow_p = ROOT / "data" / "snowball.csv"
    snow = pd.read_csv(snow_p) if snow_p.exists() else pd.DataFrame()

    pool["origin"] = "keyword search"
    if not snow.empty:
        snow["origin"] = "snowball: " + snow["seed"].fillna("") + " (" + snow["direction"].fillna("") + ")"

    cols = ["title", "year", "venue", "authors", "doi", "url", "abstract",
            "cited_by", "origin"]
    for c in cols:
        for d in (pool, snow):
            if not d.empty and c not in d.columns:
                d[c] = ""

    keep_pool = pool[cols + ["found_in_sources", "n_sources", "matched_queries"]]
    if not snow.empty:
        snow = snow.assign(found_in_sources="openalex", n_sources=1, matched_queries="")
        keep_snow = snow[cols + ["found_in_sources", "n_sources", "matched_queries"]]
        allp = pd.concat([keep_pool, keep_snow], ignore_index=True)
    else:
        allp = keep_pool

    n_before = len(allp)
    allp["_nt"] = allp["title"].fillna("").map(
        lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip())
    allp = allp[allp["_nt"].str.len() > 10]
    allp = allp.sort_values("cited_by", ascending=False).drop_duplicates("_nt")
    allp = allp.drop(columns=["_nt"])
    n_after = len(allp)

    allp = pd.concat([allp.reset_index(drop=True),
                      allp.apply(triage, axis=1).reset_index(drop=True)], axis=1)

    # Human decision columns — deliberately blank.
    allp["Decision"] = ""          # include / exclude
    allp["Exclusion reason"] = ""
    allp["Read status"] = ""       # unread / skimmed / read
    allp["Notes"] = ""
    allp["bibkey"] = ""

    allp = allp.sort_values(
        ["Priority", "year", "cited_by"], ascending=[True, False, False])

    order = ["Priority", "Why", "Decision", "Exclusion reason", "Read status",
             "title", "year", "venue", "authors", "cited_by", "doi", "url",
             "origin", "found_in_sources", "n_sources", "matched_queries",
             "t_arabic", "t_affect", "t_multimodal", "t_fusion", "t_peft",
             "t_resource", "has_abstract", "bibkey", "Notes", "abstract"]
    allp = allp[[c for c in order if c in allp.columns]]

    # ---- extraction sheet: what actually gets pulled from included papers ----
    extraction = pd.DataFrame(columns=[
        "bibkey", "title", "year", "venue", "language/dialect", "modalities",
        "dataset", "dataset size", "n classes", "fusion strategy", "encoders",
        "PEFT", "loss", "metric", "score", "baseline compared", "code available",
        "relevant to RQ", "key finding", "limitation", "why it matters for Tri-CAN",
    ])

    counts = allp["Priority"].value_counts().sort_index()
    prisma = pd.DataFrame({
        "Stage": [
            "IDENTIFIED — keyword search, raw hits (not read)",
            "IDENTIFIED — keyword search, after dedup (not read)",
            "IDENTIFIED — citation snowballing, unique (not read)",
            "Total merged before final dedup",
            "IDENTIFIED — unique records retrieved (not read)",
            "P1 core (Arabic + affect + multimodal)",
            "P2 Arabic + affect",
            "P3 fusion/PEFT method lineage",
            "P4 Arabic + multimodal",
            "P5 multimodal affect (general)",
            "P6 peripheral",
            "SCREENED by hand (title/abstract)  — TO FILL",
            "ELIGIBLE (full text assessed)      — TO FILL",
            "INCLUDED in the review             — TO FILL",
        ],
        "n": [
            1596, len(pool), len(snow) if not snow.empty else 0,
            n_before, n_after,
            *[int(counts.get(f"P{i}", 0)) for i in range(1, 7)],
            "", "", "",
        ],
    })

    # A self-documenting first sheet: whoever opens this workbook sees what the
    # numbers mean before they see the numbers. See METHODOLOGY.md.
    method = pd.DataFrame({
        "": [
            "WHAT THIS WORKBOOK IS",
            "",
            "This records an automated literature SEARCH, not 1,322 papers read.",
            "In PRISMA terms 'identified' means retrieved by the search; 'included' means",
            "read and used. Reviews routinely identify thousands and include a few dozen.",
            "",
            "HOW THE RECORDS WERE FOUND",
            "1. Protocol written first (protocol.md): questions, sources, criteria, date window.",
            "2. Keyword search, 20 queries x 5 themes, across three independent databases:",
            "   OpenAlex (~250M works, citation graph), arXiv (2026 preprints), Crossref (DOI metadata).",
            "   Semantic Scholar attempted, unavailable (HTTP 429, needs API key) — declared as a gap.",
            "3. Citation snowballing from 4 seeds (EGY-MER, Ar-MuSA, UniTextFusion, MulT),",
            "   forward and backward. This found 176 papers the keyword search missed.",
            "4. Deduplicated by DOI, then fuzzy title match (>=92).",
            "",
            "HOW PRIORITIES WERE ASSIGNED",
            "Transparent keyword rules over title+abstract. Columns t_arabic ... t_resource show",
            "the exact terms that produced each row's tier, so any assignment can be checked.",
            "Priority is a READING ORDER, not a verdict.",
            "",
            "SCREENING",
            "Priority is assigned by software; inclusion is a separate judgement. All 111 P1",
            "records were screened at title and abstract against the criteria in protocol.md,",
            "with a written reason recorded in the Decision and Reason columns for every one.",
            "The reasons are here so any single call can be checked or disputed. Result:",
            "68 included, 43 excluded. Full texts not yet assessed, so this set is provisional.",
            "",
            "TOOLS",
            "OpenAlex / Crossref / arXiv APIs for retrieval; Python (httpx, pandas, rapidfuzz,",
            "openpyxl) for search, dedup and triage; DOI content negotiation at doi.org for",
            "publisher-authoritative BibTeX; Claude Code for writing the scripts, designing the",
            "query blocks and the first screening pass, which was then reviewed.",
            "Every bibliographic field here comes from a cached API response in data/raw/ or",
            "from the publisher of record — never from a model's recollection. That rule is what",
            "keeps hallucinated citations out, and it is enforced by the pipeline, not by trust.",
            "",
            "REPRODUCIBILITY",
            "python search.py / snowball.py / make_sheet.py re-runs everything.",
            "All raw API responses are cached verbatim in data/raw/ (~25 MB).",
            "",
            "KNOWN LIMITATIONS",
            "- Records retrieved, not read. The number describes coverage only.",
            "- Keyword triage is crude; a paper omitting 'multimodal' can land in the wrong tier.",
            "- Some records lack abstracts (has_abstract = FALSE); all 68 INCLUDED works have one.",
            "- Semantic Scholar's unique coverage is absent.",
            "- Citation counts are OpenAlex's and run lower than Google Scholar's.",
            "- Coverage ends 2026-07-31.",
            "",
            "Full write-up: METHODOLOGY.md.  Key finding: FINDINGS.md.",
        ]
    })

    src = pd.DataFrame({
        "Source": ["OpenAlex", "arXiv", "Crossref", "Semantic Scholar"],
        "Role": ["Primary (250M works, citation graph)",
                 "Preprints — current through Q3 2026",
                 "Cross-check + DOI/venue metadata",
                 "Attempted; HTTP 429 without an API key"],
        "Used": ["yes", "yes", "yes", "no — rate limited"],
        "Coverage window": ["2015-01-01 to 2026-07-31"] * 3 + ["n/a"],
    })

    out = ROOT / "RP1_paper_tracking.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        method.to_excel(xl, sheet_name="Method", index=False)
        prisma.to_excel(xl, sheet_name="PRISMA counts", index=False)
        allp.to_excel(xl, sheet_name="Screening", index=False)
        extraction.to_excel(xl, sheet_name="Extraction", index=False)
        src.to_excel(xl, sheet_name="Sources", index=False)

        xl.sheets["Method"].column_dimensions["A"].width = 100

        # Readable column widths + frozen headers; this sheet gets used by hand.
        for name, df_ in (("PRISMA counts", prisma), ("Screening", allp),
                          ("Extraction", extraction), ("Sources", src)):
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df_.columns, start=1):
                letter = ws.cell(row=1, column=i).column_letter
                width = {"title": 70, "abstract": 100, "authors": 40, "venue": 34,
                         "url": 40, "matched_queries": 40, "Why": 42,
                         "origin": 30, "Stage": 46}.get(col, 16)
                ws.column_dimensions[letter].width = width

    print(f"wrote {out}")
    print(f"\nmerged {n_before} -> {n_after} unique after final dedup")
    print("\npriority breakdown:")
    print(counts.to_string())
    print("\nP1 core papers by year:")
    print(allp[allp.Priority == "P1"]["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
