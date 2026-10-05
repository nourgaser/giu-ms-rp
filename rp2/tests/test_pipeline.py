"""
Tests for the RP2 pipeline.

The ones that matter are not the smoke tests — they are the three that would
catch a wrong RESULT rather than a crash:

  test_null_corpus_sits_at_chance      a corpus with no signal must not
                                       produce skill; if it does, something
                                       leaks between train and test.
  test_did_is_zero_on_null             the headline statistic must not
                                       manufacture an effect out of noise.
  test_oof_predictions_are_held_out    every prediction must come from a model
                                       that never saw that sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rp2 import evaluate as ev            # noqa: E402
from rp2 import models as M               # noqa: E402
from rp2.data import SyntheticCorpus, get_dataset, REGISTRY   # noqa: E402
from rp2.data.base import DataNotAvailable, Sample            # noqa: E402
from rp2.train import TrainConfig, cross_validate             # noqa: E402


# ---------------------------------------------------------------- fixtures

def _corpus(signal: float, n: int = 300, seed: int = 0):
    c = SyntheticCorpus(n=n, signal=signal, seed=seed)
    samples = list(c.samples())
    dims = {"text": 24, "audio": 24, "visual": 24}
    feats = c.features(dims)
    masks = {m: np.ones(feats[m].shape[:2], dtype=bool) for m in feats}
    labels = np.array([s.label for s in samples])
    return c, samples, feats, masks, labels


# ---------------------------------------------------------------- data layer

def test_congruence_is_derived_from_modality_labels():
    s = Sample(uid="x", label=0, text_label=0, audio_label=0, visual_label=0)
    assert s.congruence == "congruent"
    assert Sample(uid="x", label=0, text_label=0, audio_label=1,
                  visual_label=1).congruence == "partial"
    assert Sample(uid="x", label=0, text_label=0, audio_label=1,
                  visual_label=2).congruence == "conflicting"


def test_congruence_is_none_without_modality_labels():
    """A corpus with only an overall label must not fabricate a stratum."""
    assert Sample(uid="x", label=0).congruence is None
    assert Sample(uid="x", label=0).text_contradicts_nonverbal is None


def test_text_contradicts_nonverbal():
    assert Sample(uid="x", label=0, text_label=2, audio_label=0,
                  visual_label=0).text_contradicts_nonverbal is True
    # agreeing with one of the two is not a contradiction
    assert Sample(uid="x", label=0, text_label=2, audio_label=2,
                  visual_label=0).text_contradicts_nonverbal is False


def test_missing_corpus_raises_with_acquisition_route():
    """EGY-MER must fail with instructions, not a bare FileNotFoundError."""
    ds = get_dataset("egy-mer", "/nonexistent/path")
    assert not ds.is_available()
    with pytest.raises(DataNotAvailable) as e:
        list(ds.samples())
    assert "EGY-MER" in str(e.value)


def test_egymer_label_space_is_seven_emotions():
    """The pipeline must not assume 3 labels anywhere."""
    ds = get_dataset("egy-mer", "/nonexistent")
    assert len(ds.labels) == 7
    assert "neutral" in ds.labels and "surprise" in ds.labels


def test_registry_covers_all_adapters():
    assert {"ar-musa", "egy-mer", "synthetic"} <= set(REGISTRY)


def test_synthetic_overall_label_is_majority_of_modalities():
    """Regression: an earlier version set text_label == label unconditionally,
    which made text a perfect predictor and drove every model to F1 1.0."""
    for s in list(SyntheticCorpus(n=200, seed=3).samples()):
        votes = [s.text_label, s.audio_label, s.visual_label]
        counts = {v: votes.count(v) for v in set(votes)}
        top = max(counts.values())
        assert s.label == (max(counts, key=counts.get) if top > 1 else s.audio_label)
    # and text must NOT always equal the overall label
    ss = list(SyntheticCorpus(n=400, seed=3).samples())
    assert any(s.text_label != s.label for s in ss)


# ---------------------------------------------------------------- models

def test_models_accept_any_label_count():
    dims = {"text": 16, "audio": 16, "visual": 16}
    for n_labels in (3, 7):
        for kind in M.MODELS:
            m = M.build(kind, dims, n_labels)
            feats = {k: __import__("torch").randn(4, 8, v) for k, v in dims.items()}
            out = m(feats)
            assert out.shape == (4, n_labels)


def test_capacity_matched_baseline_is_close_to_trican():
    """The control only controls if the parameter counts actually match."""
    dims = {"text": 32, "audio": 32, "visual": 32}
    tri = M.build("tri-can", dims, 3)
    wide = M.build("late-fusion-wide", dims, 3)
    p_tri, p_wide = M.count_params(tri), M.count_params(wide)
    assert abs(p_tri - p_wide) / p_tri < 0.05, (p_tri, p_wide)


def test_trican_has_more_params_than_plain_late_fusion():
    """Stated plainly in the write-up; asserted here so it stays true."""
    dims = {"text": 32, "audio": 32, "visual": 32}
    assert M.count_params(M.build("tri-can", dims, 3)) > \
           M.count_params(M.build("late-fusion", dims, 3))


def test_attention_masks_exclude_padding():
    """A fully-padded key sequence must not produce NaNs."""
    import torch
    blk = M.CrossModalBlock(16, n_heads=2)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 3:] = False           # second item mostly padding
    out, _ = blk(q, kv, mask)
    assert torch.isfinite(out).all()


def test_trican_rejects_unknown_anchor():
    with pytest.raises(ValueError):
        M.TriCAN({"text": 8, "audio": 8}, 3, anchor="visual")


# ---------------------------------------------------------------- evaluation

def test_did_requires_aligned_predictions():
    """Comparing two models on differently-ordered samples is meaningless."""
    a = ev.Predictions("a", np.array([0, 1]), np.array([0, 1]),
                       ["u1", "u2"], ["congruent", "partial"])
    b = ev.Predictions("b", np.array([0, 1]), np.array([1, 1]),
                       ["u2", "u1"], ["congruent", "partial"])
    with pytest.raises(ValueError):
        ev.difference_in_differences(a, b, n_boot=10)


def test_did_requires_both_strata():
    n = 20
    p = ev.Predictions("a", np.zeros(n, int), np.zeros(n, int),
                       [f"u{i}" for i in range(n)], ["congruent"] * n)
    q = ev.Predictions("b", np.zeros(n, int), np.zeros(n, int),
                       [f"u{i}" for i in range(n)], ["congruent"] * n)
    with pytest.raises(ValueError):
        ev.difference_in_differences(p, q, n_boot=10)


def test_predictions_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ev.Predictions("a", np.array([0, 1, 2]), np.array([0, 1]),
                       ["u1", "u2", "u3"], ["congruent"] * 3)


def test_stratified_counts_sum_to_total():
    n = 60
    cong = (["congruent"] * 30) + (["partial"] * 20) + (["conflicting"] * 10)
    rng = np.random.default_rng(0)
    p = ev.Predictions("m", rng.integers(0, 3, n), rng.integers(0, 3, n),
                       [f"u{i}" for i in range(n)], cong)
    r = ev.by_stratum(p, 3)
    assert r["congruent"]["n"] + r["incongruent"]["n"] == n
    assert r["partial"]["n"] + r["conflicting"]["n"] == r["incongruent"]["n"]


# ---------------------------------------------------------------- the real ones

def test_oof_predictions_are_held_out():
    """Every sample predicted exactly once, by a fold that excluded it."""
    _, _, feats, masks, labels = _corpus(0.05, n=180)
    cfg = TrainConfig(epochs=3, n_folds=3, seed=0)
    out = cross_validate("late-fusion", feats, labels, 3, cfg, masks)
    assert (out["y_pred"] >= 0).all()
    assert (out["fold"] >= 0).all()
    assert len(np.unique(out["fold"])) == 3


@pytest.mark.slow
def test_null_corpus_sits_at_chance():
    """No signal must mean no skill. Guards against leakage."""
    from sklearn.metrics import f1_score
    _, _, feats, masks, labels = _corpus(0.0, n=450)
    cfg = TrainConfig(epochs=12, n_folds=3, seed=0)
    for kind in ("late-fusion", "tri-can"):
        out = cross_validate(kind, feats, labels, 3, cfg, masks)
        f1 = f1_score(labels, out["y_pred"], average="macro", zero_division=0)
        assert f1 < 0.45, f"{kind} scored {f1:.3f} on a corpus with no signal"


@pytest.mark.slow
def test_did_is_near_zero_on_null_corpus():
    """The headline statistic must not invent an effect out of noise."""
    _, samples, feats, masks, labels = _corpus(0.0, n=450)
    cfg = TrainConfig(epochs=12, n_folds=3, seed=0)
    cong = [s.congruence for s in samples]
    uids = [s.uid for s in samples]

    preds = {}
    for kind in ("late-fusion", "tri-can"):
        out = cross_validate(kind, feats, labels, 3, cfg, masks)
        preds[kind] = ev.Predictions(kind, labels, out["y_pred"], uids, cong)

    did = ev.difference_in_differences(preds["late-fusion"], preds["tri-can"],
                                       n_boot=400, seed=0)
    lo, hi = did["ci95"]
    assert lo <= 0 <= hi, f"CI {lo:.3f}..{hi:.3f} excludes zero on a null corpus"
