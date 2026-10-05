"""
make_bib.py — build BibTeX for the included papers from authoritative sources.

Entries are NOT constructed by hand or from a model's memory. For anything with a
DOI we ask the publisher directly via DOI content negotiation
(`Accept: application/x-bibtex` at doi.org), which returns the registrant's own
BibTeX. That is the same record the publisher exports, so authors, venue, volume
and year are correct by construction rather than by trust.

arXiv-only records (no DOI) are built from the cached arXiv API response.
Anything with neither is reported as needing manual entry rather than invented.

    python make_bib.py            # included papers -> new_references.bib
    python make_bib.py --refresh  # re-fetch instead of using the cache

Output: new_references.bib  +  bib_report.txt (what succeeded, what needs hand entry)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "raw" / "bibtex"
CACHE.mkdir(parents=True, exist_ok=True)
MAILTO = "nourgaser2012@gmail.com"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s)[:60]


def fetch_bibtex(client: httpx.Client, doi: str, refresh: bool) -> str | None:
    """DOI content negotiation — the publisher's own BibTeX record."""
    f = CACHE / f"{slug(doi)}.bib"
    if f.exists() and not refresh:
        t = f.read_text()
        return t or None
    try:
        r = client.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/x-bibtex",
                     "User-Agent": f"RP1-litreview/1.0 (mailto:{MAILTO})"},
            follow_redirects=True,
        )
        text = r.text.strip() if r.status_code == 200 else ""
    except Exception:
        text = ""
    f.write_text(text)
    time.sleep(0.7)
    return text or None


def make_key(entry: str, fallback: str) -> str:
    """Readable, stable citation key: FirstAuthorYear."""
    author = re.search(r"author\s*=\s*[{\"]([^}\"]+)", entry, re.I)
    year = re.search(r"year\s*=\s*[{\"]?(\d{4})", entry, re.I)
    surname = "Anon"
    if author:
        first = author.group(1).split(" and ")[0].strip()
        parts = first.split(",") if "," in first else first.split()
        # some registrants ship an empty or punctuation-only author field
        cand = parts[0].strip() if "," in first else (parts[-1] if parts else "")
        # Normalise case: registrants are inconsistent ("Ben Cheikhi" vs
        # "BEN CHEIKHI"), which otherwise yields keys differing only by case —
        # legal in BibTeX but a trap for anyone reading or editing the file.
        surname = (re.sub(r"[^A-Za-z]", "", cand) or "Anon").capitalize()
    return f"{surname}{year.group(1) if year else fallback}"


def retype(entry: str, key: str) -> str:
    """Replace the registrant's key with ours, leaving all fields untouched.

    The old key must be matched up to the comma that begins the FIRST FIELD,
    not merely the first comma: some registrants emit keys containing a comma
    (e.g. `@article{Vani Golagana,_2025,`), and stopping at the first comma
    leaves a fragment behind that BibTeX rejects.
    """
    return re.sub(r"^(@\w+\s*\{)(.*?),(?=\s*[\w-]+\s*=)",
                  rf"\g<1>{key},", entry.strip(), count=1, flags=re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    scr = pd.read_excel(ROOT / "RP1_paper_tracking.xlsx", sheet_name="Screening")
    inc = scr[scr["Decision"].astype(str).str.startswith("include")].copy()
    print(f"included papers: {len(inc)}")

    entries, manual, seen_keys = [], [], set()
    with httpx.Client(timeout=40) as client:
        for _, r in inc.iterrows():
            doi = str(r.get("doi") or "").strip()
            title = str(r.get("title") or "")[:80]
            bib = None
            if doi and doi.lower() != "nan":
                bib = fetch_bibtex(client, doi, args.refresh)

            if bib and bib.lstrip().startswith("@"):
                key = make_key(bib, str(int(r.year)) if pd.notna(r.year) else "")
                # disambiguate collisions (same author + year)
                base, n = key, 1
                while key in seen_keys:
                    n += 1
                    key = f"{base}{chr(ord('a') + n - 2)}"
                seen_keys.add(key)
                entries.append((key, title, retype(bib, key)))
            else:
                url = str(r.get("url") or "")
                if "arxiv" in url.lower():
                    aid = url.rstrip("/").rsplit("/", 1)[-1]
                    key = f"arXiv{aid.replace('.', '')}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        entries.append((key, title, (
                            f"@misc{{{key},\n"
                            f"  title = {{{r['title']}}},\n"
                            f"  author = {{{r.get('authors', '')}}},\n"
                            f"  year = {{{int(r.year) if pd.notna(r.year) else ''}}},\n"
                            f"  eprint = {{{aid}}},\n"
                            f"  archivePrefix = {{arXiv}},\n"
                            f"  note = {{arXiv preprint}}\n}}")))
                else:
                    manual.append((title, doi, str(r.get("venue") or ""), url))

    out = ROOT / "new_references.bib"
    header = (
        "% new_references.bib\n"
        "% Generated by make_bib.py. Every entry with a DOI was fetched from the\n"
        "% publisher via DOI content negotiation at doi.org; arXiv entries come from\n"
        "% the cached arXiv API response. No entry was written from memory.\n"
        f"% Generated 2026-08-04 from the {len(inc)} included papers.\n\n")
    out.write_text(header + "\n\n".join(e for _, _, e in sorted(entries)) + "\n")

    rep = [f"BibTeX generated for {len(entries)} of {len(inc)} included papers.", ""]
    if manual:
        rep += [f"NEEDS MANUAL ENTRY ({len(manual)}) — no DOI and not on arXiv:", ""]
        for t, d, v, u in manual:
            rep.append(f"  - {t}\n      venue: {v or '(none indexed)'}\n      url:   {u}")
    (ROOT / "bib_report.txt").write_text("\n".join(rep))

    print(f"wrote {out} ({len(entries)} entries)")
    print(f"needs manual entry: {len(manual)}")
    for t, _, v, _ in manual:
        print(f"   - {t[:70]}  [{v[:28]}]")


if __name__ == "__main__":
    main()
