"""
snowball.py — citation snowballing from confirmed seed papers (protocol.md § Snowballing).

Keyword search misses work that uses different vocabulary; citation graph traversal
does not. This is the step that answers "are you sure you didn't miss anything?"

  backward — everything a seed cites   (builds the lineage behind the method)
  forward  — everything citing a seed  (surfaces the newest work, incl. 2026)

    python snowball.py [--refresh]

Output: data/snowball.csv, and a coverage report against data/pool.csv.
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
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

MAILTO = "nourgaser2012@gmail.com"
UA = f"RP1-litreview/1.0 (mailto:{MAILTO})"

# Seeds confirmed to resolve in OpenAlex (see protocol.md).
SEEDS = {
    "W4416420869": "EGY-MER (Hussein et al. 2025)",
    "W4409406878": "Ar-MuSA (Khaled et al. 2025)",
    "W4413353552": "UniTextFusion (Khaled et al. 2025)",
    "W2947476638": "MulT (Tsai et al. 2019)",
}


def _cache(name: str, fetch, refresh: bool):
    p = RAW / f"{name}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    data = fetch()
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    time.sleep(1.0)
    return data


def _invert(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def _row(w, seed_label, direction):
    return {
        "source": "openalex-snowball",
        "source_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "title": w.get("display_name") or "",
        "year": w.get("publication_year"),
        "venue": (((w.get("primary_location") or {}).get("source") or {}) or {}).get("display_name") or "",
        "authors": "; ".join(
            (a.get("author") or {}).get("display_name", "")
            for a in (w.get("authorships") or [])[:8]
        ),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "url": w.get("id") or "",
        "abstract": _invert(w.get("abstract_inverted_index")),
        "cited_by": w.get("cited_by_count"),
        "seed": seed_label,
        "direction": direction,
    }


def forward(client, wid, label, refresh):
    """Works citing the seed. Paginated — seeds like MulT have many."""
    rows, cursor, page = [], "*", 0
    while cursor and page < 6:
        def fetch():
            r = client.get(
                "https://api.openalex.org/works",
                params={"filter": f"cites:{wid}", "per-page": 200,
                        "cursor": cursor, "mailto": MAILTO},
            )
            r.raise_for_status()
            return r.json()

        data = _cache(f"snow_fwd__{wid}__p{page}", fetch, refresh)
        for w in data.get("results", []):
            rows.append(_row(w, label, "forward"))
        cursor = (data.get("meta") or {}).get("next_cursor")
        page += 1
        if not data.get("results"):
            break
    return rows


def backward(client, wid, label, refresh):
    """Works the seed cites. Fetched in OR-batches of 50 to limit requests."""
    def fetch_seed():
        r = client.get(f"https://api.openalex.org/works/{wid}",
                       params={"mailto": MAILTO})
        r.raise_for_status()
        return r.json()

    seed = _cache(f"snow_seedmeta__{wid}", fetch_seed, refresh)
    refs = [r.rsplit("/", 1)[-1] for r in (seed.get("referenced_works") or [])]
    rows = []
    for i in range(0, len(refs), 50):
        batch = refs[i:i + 50]

        def fetch():
            r = client.get(
                "https://api.openalex.org/works",
                params={"filter": f"openalex_id:{'|'.join(batch)}",
                        "per-page": 200, "mailto": MAILTO},
            )
            r.raise_for_status()
            return r.json()

        data = _cache(f"snow_bwd__{wid}__b{i//50}", fetch, refresh)
        for w in data.get("results", []):
            rows.append(_row(w, label, "backward"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rows = []
    with httpx.Client(timeout=45, headers={"User-Agent": UA}) as client:
        for wid, label in SEEDS.items():
            f = forward(client, wid, label, args.refresh)
            b = backward(client, wid, label, args.refresh)
            rows.extend(f + b)
            print(f"  {label:34s} forward={len(f):4d}  backward={len(b):4d}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("no snowball results")
        return

    df["_nt"] = df["title"].map(lambda t: re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip())
    df = df.drop_duplicates(subset=["_nt"]).drop(columns=["_nt"])
    out = ROOT / "data" / "snowball.csv"
    df.to_csv(out, index=False)
    print(f"\nunique snowball records: {len(df)} -> {out}")

    # How much did snowballing add that keyword search missed? This number is
    # the justification for doing it at all, and belongs in the write-up.
    pool_p = ROOT / "data" / "pool.csv"
    if pool_p.exists():
        pool = pd.read_csv(pool_p)
        known = set(pool["title"].fillna("").map(
            lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()))
        new = df[~df["title"].fillna("").map(
            lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()).isin(known)]
        print(f"NOT already in keyword pool: {len(new)}")
        print("\nby year (new only):")
        print(new["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
