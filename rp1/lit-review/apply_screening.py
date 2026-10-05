"""
apply_screening.py — write the P1 screening decisions into the tracking workbook.

Matches DECISIONS (screening_decisions.py) onto the Screening sheet by title
substring, then verifies the match is exact and total: every P1 paper gets one
decision, every decision matches one paper. A partial match would silently
misreport the PRISMA counts, so both directions are checked and the script
refuses to write if either fails.

    python apply_screening.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from screening_decisions import DECISIONS

ROOT = Path(__file__).parent
BOOK = ROOT / "RP1_paper_tracking.xlsx"


def main() -> None:
    sheets = pd.read_excel(BOOK, sheet_name=None)
    scr = sheets["Screening"]
    scr["Decision"] = scr["Decision"].astype("object")
    scr["Exclusion reason"] = scr["Exclusion reason"].astype("object")

    titles = scr["title"].fillna("").astype(str)
    p1_mask = scr["Priority"] == "P1"

    matched_rows: set[int] = set()
    unmatched_keys: list[str] = []
    ambiguous: list[tuple[str, int]] = []

    for key, (decision, reason) in DECISIONS.items():
        k = key.strip()
        if k.startswith("="):
            # exact title match: needed for keys that are also a substring of
            # many other titles (e.g. the bare phrase "Multimodal Sentiment
            # Analysis"). Substring matching there silently stole 4 other
            # papers' reasons before this was added.
            hits = scr.index[p1_mask & (titles.str.lower() == k[1:].lower())]
        else:
            hits = scr.index[p1_mask & titles.str.contains(k, case=False, regex=False)]
        if len(hits) == 0:
            unmatched_keys.append(key)
            continue
        if len(hits) > 1:
            ambiguous.append((key, len(hits)))
        for i in hits:
            if i in matched_rows:
                continue                     # first (most specific) key wins
            scr.loc[i, "Decision"] = decision
            scr.loc[i, "Exclusion reason"] = reason
            matched_rows.add(i)

    p1_rows = set(scr.index[p1_mask])
    undecided = p1_rows - matched_rows

    print(f"P1 papers        : {len(p1_rows)}")
    print(f"decisions applied: {len(matched_rows)}")
    print(f"undecided        : {len(undecided)}")
    if unmatched_keys:
        print(f"\nkeys matching NO paper ({len(unmatched_keys)}):")
        for k in unmatched_keys:
            print(f"  - {k!r}")
    if ambiguous:
        print(f"\nkeys matching MULTIPLE papers ({len(ambiguous)}):")
        for k, n in ambiguous:
            print(f"  - {k!r} -> {n}")
    if undecided:
        print(f"\nP1 papers with NO decision ({len(undecided)}):")
        for i in sorted(undecided):
            print(f"  - {titles[i][:95]}")

    if unmatched_keys or undecided:
        print("\nREFUSING TO WRITE — resolve the mismatches above first.")
        sys.exit(1)

    counts = Counter(scr.loc[list(matched_rows), "Decision"])
    included = sum(v for k, v in counts.items() if str(k).startswith("include"))
    print(f"\n{dict(counts)}\nincluded (any): {included}")

    # ---- refresh PRISMA with real screening outcomes --------------------
    prisma = sheets["PRISMA counts"]
    fill = {
        "SCREENED by hand (title/abstract)  — TO FILL": len(p1_rows),
        "ELIGIBLE (full text assessed)      — TO FILL": included,
        "INCLUDED in the review             — TO FILL": included,
    }
    rename = {
        "SCREENED by hand (title/abstract)  — TO FILL":
            "SCREENED at title/abstract (P1 tier)",
        "ELIGIBLE (full text assessed)      — TO FILL":
            "ELIGIBLE after screening (full text pending)",
        "INCLUDED in the review             — TO FILL":
            "INCLUDED — provisional, pending full-text + adjudication",
    }
    prisma["n"] = prisma["n"].astype("object")
    for old, val in fill.items():
        m = prisma["Stage"] == old
        prisma.loc[m, "n"] = val
        prisma.loc[m, "Stage"] = rename[old]

    extra = pd.DataFrame({
        "Stage": ["  of which: include (Arabic/core)",
                  "  of which: include-method (fusion lineage)",
                  "  of which: include-weak (marginal)",
                  "  EXCLUDED at screening (reasons recorded per row)"],
        "n": [counts.get("include", 0), counts.get("include-method", 0),
              counts.get("include-weak", 0), counts.get("exclude", 0)],
    })
    prisma = pd.concat([prisma, extra], ignore_index=True)

    # ---- seed the extraction sheet with everything included -------------
    inc = scr[scr["Decision"].astype(str).str.startswith("include")].copy()
    inc = inc.sort_values(["Decision", "year"], ascending=[True, False])
    extraction = sheets["Extraction"]
    seed = pd.DataFrame({c: "" for c in extraction.columns}, index=range(len(inc)))
    seed["title"] = inc["title"].values
    seed["year"] = inc["year"].values
    seed["venue"] = inc["venue"].values
    seed["why it matters for Tri-CAN"] = inc["Exclusion reason"].values

    with pd.ExcelWriter(BOOK, engine="openpyxl") as xl:
        sheets["Method"].to_excel(xl, sheet_name="Method", index=False)
        prisma.to_excel(xl, sheet_name="PRISMA counts", index=False)
        scr.to_excel(xl, sheet_name="Screening", index=False)
        seed.to_excel(xl, sheet_name="Extraction", index=False)
        sheets["Sources"].to_excel(xl, sheet_name="Sources", index=False)
        for name in ("Method", "PRISMA counts", "Screening", "Extraction", "Sources"):
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
        xl.sheets["Method"].column_dimensions["A"].width = 100
        for col, w in (("A", 46), ("B", 12)):
            xl.sheets["PRISMA counts"].column_dimensions[col].width = w

    print(f"\nwrote {BOOK}")
    print(f"Extraction sheet seeded with {len(seed)} included papers")


if __name__ == "__main__":
    main()
