"""
search.py — reproducible multi-source literature search for RP1.

Implements the keyword-search stage of protocol.md. Every API response is cached
verbatim under data/raw/ so the search is auditable and re-runnable offline: if a
reviewer asks "where did this paper come from?", the answer is a file.

    python search.py              # run all blocks, use cache where present
    python search.py --refresh    # ignore cache, re-query everything
    python search.py --block A    # single block

Output: data/pool.csv — deduplicated candidate pool.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from pathlib import Path

import httpx
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# OpenAlex "polite pool" — identifying yourself gets faster, more reliable service.
MAILTO = "nourgaser2012@gmail.com"
UA = f"RP1-litreview/1.0 (mailto:{MAILTO})"

# Coverage window from protocol.md.
FROM_DATE = "2015-01-01"
TO_DATE = "2026-07-31"


# --------------------------------------------------------------------------
# Query blocks (protocol.md § Search blocks)
#
# Kept as plain phrase lists so the same intent can be expressed in each API's
# own syntax below, rather than hand-writing per-source query strings.
# --------------------------------------------------------------------------
BLOCKS: dict[str, list[str]] = {
    "A": [  # Arabic multimodal affect — core
        "arabic multimodal sentiment analysis",
        "arabic multimodal emotion recognition",
        "arabic multimodal affective computing",
        "multimodal sentiment analysis low-resource language",
    ],
    "B": [  # Dialectal
        "egyptian arabic emotion recognition",
        "dialectal arabic sentiment analysis speech",
        "egyptian arabic speech emotion",
        "arabic dialect multimodal",
    ],
    "C": [  # Fusion mechanisms
        "cross-modal attention multimodal sentiment",
        "multimodal transformer unaligned sequences",
        "cross-modal attention emotion recognition",
        "late fusion versus early fusion multimodal emotion",
        "modality fusion parameter efficient LoRA multimodal",
    ],
    "D": [  # Resources
        "arabic multimodal dataset emotion",
        "arabic multimodal benchmark sentiment",
        "arabic audiovisual emotion corpus",
    ],
    "E": [  # Adjacent
        "arabic sarcasm detection multimodal",
        "arabic irony detection",
        "arabic speech emotion recognition wav2vec",
        "arabic large multimodal model vision language",
    ],
}


@dataclass
class Record:
    """Normalised across all four sources so they can be pooled and deduped."""
    source: str
    source_id: str
    title: str
    year: int | None
    venue: str
    authors: str
    doi: str
    url: str
    abstract: str
    cited_by: int | None
    block: str
    query: str
    extra: dict = field(default_factory=dict)


def _norm_title(t: str) -> str:
    """Aggressive normalisation for dedup — punctuation and spacing vary by source."""
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _cache(name: str, fetch, refresh: bool):
    """Read-through cache. Keeps raw responses for the audit trail."""
    p = RAW / f"{name}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    data = fetch()
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    time.sleep(1.0)  # be a good citizen regardless of documented limits
    return data


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def openalex(client: httpx.Client, q: str, block: str, refresh: bool) -> list[Record]:
    """Primary source. `title_and_abstract.search` is more precise than `search`."""
    def fetch():
        r = client.get(
            "https://api.openalex.org/works",
            params={
                "filter": (
                    f"title_and_abstract.search:{q},"
                    f"from_publication_date:{FROM_DATE},"
                    f"to_publication_date:{TO_DATE}"
                ),
                "per-page": 50,
                "mailto": MAILTO,
            },
        )
        r.raise_for_status()
        return r.json()

    data = _cache(f"openalex__{block}__{_slug(q)}", fetch, refresh)
    out = []
    for w in data.get("results", []):
        out.append(Record(
            source="openalex",
            source_id=(w.get("id") or "").rsplit("/", 1)[-1],
            title=w.get("display_name") or "",
            year=w.get("publication_year"),
            venue=(((w.get("primary_location") or {}).get("source") or {}) or {}).get("display_name") or "",
            authors="; ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:8]
            ),
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            url=w.get("id") or "",
            abstract=_invert_abstract(w.get("abstract_inverted_index")),
            cited_by=w.get("cited_by_count"),
            block=block,
            query=q,
            extra={"type": w.get("type")},
        ))
    return out


def _invert_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index for licensing reasons."""
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def arxiv(client: httpx.Client, q: str, block: str, refresh: bool) -> list[Record]:
    """Preprints — the only source current enough for 2026. Note: needs HTTPS
    (the http:// endpoint 301s and silently returns nothing)."""
    def fetch():
        terms = " AND ".join(f'all:"{t}"' for t in q.split()[:4])
        r = client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": terms,
                "start": 0,
                "max_results": 40,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            follow_redirects=True,
        )
        r.raise_for_status()
        return {"xml": r.text}

    data = _cache(f"arxiv__{block}__{_slug(q)}", fetch, refresh)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(data["xml"])
    except ET.ParseError:
        return []

    out = []
    for e in root.findall("a:entry", ns):
        def txt(tag):
            el = e.find(f"a:{tag}", ns)
            return (el.text or "").strip() if el is not None else ""
        aid = txt("id")
        pub = txt("published")
        doi_el = e.find("a:doi", ns)
        out.append(Record(
            source="arxiv",
            source_id=aid.rsplit("/", 1)[-1],
            title=" ".join(txt("title").split()),
            year=int(pub[:4]) if pub[:4].isdigit() else None,
            venue="arXiv preprint",
            authors="; ".join(
                (a.find("a:name", ns).text or "")
                for a in e.findall("a:author", ns)[:8]
            ),
            doi=(doi_el.text if doi_el is not None else "") or "",
            url=aid,
            abstract=" ".join(txt("summary").split()),
            cited_by=None,
            block=block,
            query=q,
            extra={"published": pub},
        ))
    return out


def crossref(client: httpx.Client, q: str, block: str, refresh: bool) -> list[Record]:
    """Cross-check + authoritative DOI/venue metadata. No abstracts."""
    def fetch():
        r = client.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": q,
                "rows": 40,
                "filter": f"from-pub-date:{FROM_DATE},until-pub-date:{TO_DATE}",
                "mailto": MAILTO,
            },
        )
        r.raise_for_status()
        return r.json()

    data = _cache(f"crossref__{block}__{_slug(q)}", fetch, refresh)
    out = []
    for it in data.get("message", {}).get("items", []):
        parts = (it.get("issued") or {}).get("date-parts") or [[None]]
        out.append(Record(
            source="crossref",
            source_id=it.get("DOI", ""),
            title=" ".join((it.get("title") or [""])[0].split()),
            year=parts[0][0] if parts and parts[0] else None,
            venue=(it.get("container-title") or [""])[0],
            authors="; ".join(
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in (it.get("author") or [])[:8]
            ),
            doi=it.get("DOI", ""),
            url=it.get("URL", ""),
            abstract=re.sub(r"<[^>]+>", "", it.get("abstract", "") or ""),
            cited_by=it.get("is-referenced-by-count"),
            block=block,
            query=q,
            extra={"type": it.get("type")},
        ))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:60].strip("-")


# --------------------------------------------------------------------------
# Dedup + assembly
# --------------------------------------------------------------------------
def dedupe(records: list[Record]) -> pd.DataFrame:
    """DOI exact-match first, then fuzzy title (>=92). Keeps the richest copy and
    records which sources agreed — cross-source agreement is a quality signal."""
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        return df
    df["doi"] = df["doi"].fillna("").str.lower().str.strip()
    df["_nt"] = df["title"].map(_norm_title)
    df = df[df["_nt"].str.len() > 10].copy()

    # richest record wins: prefer one with an abstract, then higher citation count
    df["_rank"] = df["abstract"].str.len().fillna(0) + df["cited_by"].fillna(0) * 0.01
    df = df.sort_values("_rank", ascending=False)

    keep: list[int] = []
    seen_doi: dict[str, int] = {}
    seen_titles: list[tuple[str, int]] = []
    merged_sources: dict[int, set] = {}
    merged_queries: dict[int, set] = {}

    for idx, row in df.iterrows():
        hit = None
        if row["doi"] and row["doi"] in seen_doi:
            hit = seen_doi[row["doi"]]
        else:
            for nt, kidx in seen_titles:
                if fuzz.ratio(row["_nt"], nt) >= 92:
                    hit = kidx
                    break
        if hit is None:
            keep.append(idx)
            if row["doi"]:
                seen_doi[row["doi"]] = idx
            seen_titles.append((row["_nt"], idx))
            merged_sources[idx] = {row["source"]}
            merged_queries[idx] = {f'{row["block"]}:{row["query"]}'}
        else:
            merged_sources[hit].add(row["source"])
            merged_queries[hit].add(f'{row["block"]}:{row["query"]}')

    out = df.loc[keep].copy()
    out["found_in_sources"] = out.index.map(lambda i: ", ".join(sorted(merged_sources[i])))
    out["n_sources"] = out.index.map(lambda i: len(merged_sources[i]))
    out["matched_queries"] = out.index.map(lambda i: " | ".join(sorted(merged_queries[i])))
    return out.drop(columns=["_nt", "_rank"]).sort_values(
        ["n_sources", "cited_by"], ascending=False
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    ap.add_argument("--block", help="run one block only (A-E)")
    args = ap.parse_args()

    blocks = {args.block: BLOCKS[args.block]} if args.block else BLOCKS
    records: list[Record] = []
    stats: list[tuple] = []

    with httpx.Client(timeout=45, headers={"User-Agent": UA}) as client:
        for block, queries in blocks.items():
            for q in queries:
                per_source = {}
                for name, fn in (("openalex", openalex), ("arxiv", arxiv), ("crossref", crossref)):
                    try:
                        got = fn(client, q, block, args.refresh)
                        records.extend(got)
                        per_source[name] = len(got)
                    except Exception as e:  # a dead source must not kill the run
                        per_source[name] = f"ERR {type(e).__name__}"
                stats.append((block, q, per_source))
                print(f"  [{block}] {q[:52]:52s} {per_source}")

    print(f"\nraw records: {len(records)}")
    df = dedupe(records)
    print(f"after dedup: {len(df)}")

    out = ROOT / "data" / "pool.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    if not df.empty:
        print("\nby year:")
        print(df["year"].value_counts().sort_index().to_string())
        recent = df[df["year"] >= 2026]
        print(f"\n2026 papers found: {len(recent)}")


if __name__ == "__main__":
    main()
