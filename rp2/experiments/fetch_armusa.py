"""
fetch_armusa.py — resumable download of Ar-MuSA from the Hugging Face hub.

    python -m experiments.fetch_armusa --root data/raw/ar-musa
    python -m experiments.fetch_armusa --root ... --subsample 3000

Why this does not use `huggingface_hub.snapshot_download`:

    Ar-MuSA is ~17,400 small media files. snapshot_download routes them through
    the hub's xet transfer path, which rate-limited this (unauthenticated)
    client to roughly 1-2 files per minute after the first few hundred, and
    intermittently returned HTTP 429. Measured against the plain
    `resolve/main` HTTPS endpoint the same files come down at ~4 per second
    with 20 concurrent connections — three orders of magnitude faster.

    So this fetches directly over HTTPS with a thread pool. It is resumable
    (existing non-empty files are skipped), guards the disk, and reports what
    failed rather than exiting on the first error.

`--subsample N` fetches a stratified sample instead of the whole corpus, which
is the right move when the full 10 GB is not needed: see `stratified_ids`.

Not fetched, because nothing in this project reads them: `*.docx` (~500 files)
and `*/transcripts/*.txt` (8,205 files). Both are duplicated by the CSV's
`Transcript` column, and the .txt set is 495 short of the 8,700 rows, so using
it would silently drop samples.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://huggingface.co/datasets/Skhaled/Ar-MUSA/resolve/main"
BATCHES = ("1a", "1b", "2a", "2b", "3a", "3b", "3c",
           "4a", "4b", "4c", "5a", "5b", "5c")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def stratified_ids(root: Path, n: int, seed: int = 0) -> list:
    """Pick `n` samples preserving the congruence x label distribution.

    Stratifying on (congruence, overall label) rather than sampling uniformly:
    the whole analysis is a comparison BETWEEN congruence strata, so their
    relative sizes and their internal class balance both have to survive
    subsampling. Uniform sampling preserves them only in expectation, and the
    'conflicting' stratum is just 4.7% of the corpus.
    """
    import random
    from collections import defaultdict

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rp2.data import ArMuSA

    samples = list(ArMuSA(root, require_media=False).samples())
    if n >= len(samples):
        return samples

    buckets: dict[tuple, list] = defaultdict(list)
    for s in samples:
        buckets[(s.congruence, s.label)].append(s)

    rng = random.Random(seed)
    chosen: list = []
    for _, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(group)
        chosen.extend(group[:max(round(n * len(group) / len(samples)), 1)])

    rng.shuffle(chosen)
    return chosen[:n]


def fetch_csvs(root: Path, timeout: float = 60.0) -> int:
    """The annotation CSVs — small, and they carry the labels AND transcripts.

    Fetched first and separately: with these alone the congruence statistics
    and the sampling plan can be computed before any media is downloaded.
    """
    import httpx

    got = 0
    with httpx.Client(timeout=timeout, follow_redirects=True) as c:
        for b in BATCHES:
            dest = root / f"Annotation{b}" / f"updated_annotation_{b}.csv"
            if dest.exists() and dest.stat().st_size > 0:
                got += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = c.get(f"{BASE}/Annotation{b}/updated_annotation_{b}.csv")
            if r.status_code == 200:
                dest.write_bytes(r.content)
                got += 1
            else:
                print(f"  WARN: CSV {b} -> HTTP {r.status_code}", file=sys.stderr)
    return got


def plan(root: Path, chosen: list | None) -> list[tuple[str, Path]]:
    """Build the (url, destination) work list, skipping what is already here."""
    import csv as _csv

    jobs: list[tuple[str, Path]] = []

    def add(batch: str, sid: str) -> None:
        for sub, pre, ext in (("audio_segments", "audio", "wav"),
                              ("frames", "frame", "jpg")):
            dest = root / f"Annotation{batch}" / sub / f"{pre}_{sid}.{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            jobs.append((f"{BASE}/Annotation{batch}/{sub}/{pre}_{sid}.{ext}", dest))

    if chosen is not None:
        for s in chosen:
            add(s.meta["batch"], s.uid.split(":", 1)[1])
        return jobs

    for b in BATCHES:
        csv_path = root / f"Annotation{b}" / f"updated_annotation_{b}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", errors="replace") as fh:
            for row in _csv.DictReader(fh):
                sid = (row.get("ID") or "").strip()
                if sid:
                    add(b, sid)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ar-musa"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--subsample", type=int, default=0,
                    help="fetch only N clips, stratified by congruence x label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="stop rather than fill the disk below this")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    import httpx

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"annotation CSVs: {fetch_csvs(args.root, args.timeout)}/{len(BATCHES)}")

    chosen = None
    if args.subsample:
        chosen = stratified_ids(args.root, args.subsample, args.seed)
        from collections import Counter
        print(f"stratified subsample: {len(chosen)} clips")
        print(f"  congruence: {dict(Counter(s.congruence for s in chosen))}")
        print(f"  labels    : {dict(Counter(s.label for s in chosen))}")

    jobs = plan(args.root, chosen)
    print(f"{len(jobs)} files to fetch ({free_gb(args.root):.1f} GB free)", flush=True)
    if not jobs:
        print("nothing to do — already complete")
        return 0

    done = failed = 0
    stop = threading.Event()
    lock = threading.Lock()
    started = time.time()

    def get(url: str, dest: Path) -> bool:
        if stop.is_set():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with httpx.Client(timeout=args.timeout, follow_redirects=True) as c:
                    r = c.get(url)
                if r.status_code == 200 and r.content:
                    # Write via a temp file so an interrupted run never leaves a
                    # truncated file that the resume logic would treat as done.
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(dest)
                    return True
                if r.status_code == 404:
                    return False
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get, u, d): (u, d) for u, d in jobs}
        for fut in as_completed(futures):
            ok = False
            try:
                ok = fut.result()
            except Exception:
                ok = False
            with lock:
                if ok:
                    done += 1
                else:
                    failed += 1
                n = done + failed
                if n % 250 == 0:
                    el = time.time() - started
                    rate = n / max(el, 1e-9)
                    print(f"  {n}/{len(jobs)}  {done} ok, {failed} missing  "
                          f"{rate:.1f} files/s  eta {(len(jobs)-n)/max(rate,1e-9)/60:.1f} min  "
                          f"({free_gb(args.root):.1f} GB free)", flush=True)
                    if free_gb(args.root) < args.min_free_gb:
                        print("STOPPING: disk below --min-free-gb", file=sys.stderr)
                        stop.set()

    el = time.time() - started
    n_wav = len(list(args.root.rglob("*.wav")))
    n_jpg = len(list(args.root.rglob("*.jpg")))
    size = sum(f.stat().st_size for f in args.root.rglob("*") if f.is_file())
    print(f"\n{done} fetched, {failed} missing/failed in {el/60:.1f} min")
    print(f"on disk: {n_wav} wav, {n_jpg} jpg, {size/1e9:.2f} GB")
    if stop.is_set():
        print("stopped early on the disk guard; rerun to resume", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
