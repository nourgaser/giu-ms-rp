"""
merge_bib.py — merge the newly fetched entries into rp1/report/references.bib.

The existing bibliography's keys are cited throughout main.tex, so they are
treated as immutable: where a newly fetched entry describes a paper already in
the file, the OLD key wins and the new entry is dropped. Renaming a key would
silently break a \\citep in the manuscript.

Duplicate detection is by DOI first, then normalised title.

    python merge_bib.py [--write]     # dry run unless --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).parent
EXISTING = ROOT.parent / "report" / "references.bib"
NEW = ROOT / "new_references.bib"


def split_entries(text: str) -> list[str]:
    """Split a .bib file into entries by brace balance (entries may be one line
    or many, and fields contain braces)."""
    out, i = [], 0
    while True:
        start = text.find("@", i)
        if start == -1:
            break
        depth, j, seen = 0, start, False
        while j < len(text):
            if text[j] == "{":
                depth += 1
                seen = True
            elif text[j] == "}":
                depth -= 1
                if seen and depth == 0:
                    j += 1
                    break
            j += 1
        out.append(text[start:j].strip())
        i = j
    return [e for e in out if e]


def field(entry: str, name: str) -> str:
    m = re.search(rf"{name}\s*=\s*\{{(.*?)\}}\s*[,\n}}]", entry, re.I | re.S)
    if not m:
        m = re.search(rf"{name}\s*=\s*\{{(.*?)\}}", entry, re.I | re.S)
    return (m.group(1).strip() if m else "")


def key_of(entry: str) -> str:
    m = re.match(r"@\w+\s*\{\s*([^,]+),", entry)
    return m.group(1).strip() if m else ""


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    old_text = EXISTING.read_text()
    old = split_entries(old_text)
    new = split_entries(NEW.read_text())
    print(f"existing entries: {len(old)}   fetched entries: {len(new)}")

    old_dois = {field(e, "doi").lower(): key_of(e) for e in old if field(e, "doi")}
    old_titles = {norm_title(field(e, "title")): key_of(e) for e in old if field(e, "title")}

    keep, dupes = [], []
    for e in new:
        doi, title = field(e, "doi").lower(), norm_title(field(e, "title"))
        hit = old_dois.get(doi) if doi else None
        if not hit and title:
            hit = old_titles.get(title)
        if hit:
            dupes.append((key_of(e), hit, field(e, "title")[:64]))
        else:
            keep.append(e)

    # A *different* paper by the same first author in the same year generates
    # the same key as an existing entry (e.g. two Youcef 2021 papers). BibTeX
    # then silently drops one. Rename the incoming entry, never the existing.
    taken = {key_of(e) for e in old}
    renamed = []
    for i, e in enumerate(keep):
        k = key_of(e)
        if k in taken:
            new_k, suffix = k, ord("a")
            while new_k in taken:
                new_k = f"{k}{chr(suffix)}"
                suffix += 1
            keep[i] = re.sub(r"^(@\w+\s*\{\s*)[^,]+,", rf"\g<1>{new_k},", e, count=1)
            renamed.append((k, new_k, field(e, "title")[:56]))
            taken.add(new_k)
        else:
            taken.add(k)
    if renamed:
        print(f"\nkey collisions with existing bib, incoming entry renamed ({len(renamed)}):")
        for a, b, t in renamed:
            print(f"  {a} -> {b}   {t}")

    print(f"\nalready present, dropped ({len(dupes)}) — existing key kept:")
    for newk, oldk, t in dupes:
        print(f"  {newk:>18} == {oldk:<18} {t}")
    print(f"\nnew entries to add: {len(keep)}")

    if not args.write:
        print("\n(dry run — pass --write to apply)")
        return

    merged = (old_text.rstrip()
              + "\n\n% ---------------------------------------------------------------\n"
                "% Added 2026-08-04 from the systematic search (lit-review/).\n"
                "% Fetched via DOI content negotiation; see lit-review/make_bib.py.\n"
                "% ---------------------------------------------------------------\n\n"
              + "\n\n".join(keep) + "\n")
    EXISTING.write_text(merged)
    print(f"\nwrote {EXISTING}: {len(old)} + {len(keep)} = {len(old) + len(keep)} entries")


if __name__ == "__main__":
    main()
