"""
rp2.data — corpus registry.

Adding a corpus is: write an adapter, add one line to `REGISTRY`. Nothing
else in the codebase changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .armusa import ArMuSA
from .base import ALL_MODALITIES, DataNotAvailable, DatasetAdapter, Modality, Sample
from .egymer import EgyMer
from .synthetic import SyntheticCorpus

# name -> (factory taking a root path, one-line note on how to obtain it)
REGISTRY: dict[str, tuple[Callable[[Path], object], str]] = {
    "ar-musa": (ArMuSA, "open download, HF Skhaled/Ar-MUSA (AFL-3.0)"),
    "egy-mer": (EgyMer, "obtained 2026-08-09; 484 usable of 5,471 segments"),
    "synthetic": (lambda root=None: SyntheticCorpus(), "generated, no data needed"),
}


def get_dataset(name: str, root: str | Path | None = None):
    if name not in REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}. known: {', '.join(sorted(REGISTRY))}"
        )
    factory, _ = REGISTRY[name]
    return factory(Path(root)) if root is not None else factory()


def describe() -> str:
    """Human-readable availability report — used by `rp2-run --list`."""
    lines = []
    for name, (_, note) in sorted(REGISTRY.items()):
        lines.append(f"  {name:12s}  {note}")
    return "\n".join(lines)


__all__ = [
    "ALL_MODALITIES", "ArMuSA", "DataNotAvailable", "DatasetAdapter", "EgyMer",
    "Modality", "REGISTRY", "Sample", "SyntheticCorpus", "describe", "get_dataset",
]
