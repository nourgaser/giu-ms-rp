"""
evaluate.py — metrics, and the congruence-stratified comparison.

The headline number of this study is NOT "our F1 on Ar-MuSA". Three 2026
papers already compete on that benchmark and the best of them reports 0.8989;
entering that race with a frozen-encoder CPU model would place fourth and prove
nothing.

The headline is a WITHIN-STUDY CONTRAST:

    Does the advantage of cross-modal attention over late fusion depend on
    whether the modalities agree with each other?

Ar-MuSA annotates sentiment separately for transcript, audio and frame, so
"the modalities disagree" is a property of the corpus's own labels, fixed
before any model is trained. 39.0% of the 8,700 samples are congruent (all
three annotations identical), 61.0% are not. That split is not chosen by us
and cannot be tuned.

Why the contrast is more defensible than a leaderboard number:

  * It is immune to absolute performance. If both models are mediocre, the
    difference between strata is still meaningful.
  * It is directly motivated by the architecture. Cross-modal attention exists
    to let one modality condition on another; if it helps anywhere, it should
    help most where the modalities carry conflicting evidence. A null result
    here is evidence AGAINST the architectural story and is reported as such.
  * Nobody has published it. The three Ar-MuSA papers report aggregate F1 only.

Statistical care taken here, because this is where a study like this usually
falls over:

  * The strata have different sizes and different class distributions, so raw
    F1 is not comparable ACROSS strata. Only the model-vs-model gap WITHIN a
    stratum is interpreted, and the difference of those gaps is the estimand.
  * The estimand is a difference-in-differences, and its uncertainty is
    estimated by bootstrapping over samples within folds, not by comparing
    two means with independent error bars.
  * Paired comparisons throughout: both models see identical folds and
    identical samples, so the pairing is real and is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

STRATA = ("congruent", "partial", "conflicting")


@dataclass
class Predictions:
    """Predictions for one model on one evaluation set."""

    model: str
    y_true: np.ndarray
    y_pred: np.ndarray
    uids: list[str]
    congruence: list[str | None]
    sarcasm: list[int | None] = field(default_factory=list)
    fold: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true)
        self.y_pred = np.asarray(self.y_pred)
        n = len(self.y_true)
        if not (len(self.y_pred) == len(self.uids) == len(self.congruence) == n):
            raise ValueError("Predictions: array lengths disagree")


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> dict:
    """Macro-F1 is the headline; weighted-F1 is reported for comparability.

    Macro because the corpora are imbalanced and macro refuses to be flattered
    by the majority class. Weighted as well, because the published EGY-MER
    baseline (0.519) is a weighted figure and a comparison against it has to be
    like-for-like.
    """
    if len(y_true) == 0:
        return {"n": 0, "macro_f1": float("nan"), "weighted_f1": float("nan"),
                "accuracy": float("nan")}
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(n_labels))).tolist(),
    }


def by_stratum(p: Predictions, n_labels: int) -> dict[str, dict]:
    """Metrics computed separately within each congruence stratum."""
    cong = np.array([c if c is not None else "unknown" for c in p.congruence])
    out = {"overall": core_metrics(p.y_true, p.y_pred, n_labels)}
    for s in STRATA:
        m = cong == s
        if m.any():
            out[s] = core_metrics(p.y_true[m], p.y_pred[m], n_labels)
    # The two-way split the argument actually rests on.
    m = cong == "congruent"
    if m.any() and (~m).any():
        out["incongruent"] = core_metrics(p.y_true[~m], p.y_pred[~m], n_labels)
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def difference_in_differences(
    baseline: Predictions,
    proposed: Predictions,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """The main estimand, with a bootstrap interval.

        gap_incongruent = F1(proposed | incongruent) - F1(baseline | incongruent)
        gap_congruent   = F1(proposed | congruent)   - F1(baseline | congruent)
        DiD             = gap_incongruent - gap_congruent

    DiD > 0 means cross-modal attention helps MORE where the modalities
    disagree, which is what the architecture predicts. DiD ~ 0 means whatever
    advantage it has is uniform, and the "it resolves conflicting evidence"
    story is not supported — that is a real finding and gets reported plainly.

    Resampling is over samples, stratified so each bootstrap replicate keeps
    the two strata at their observed sizes. Both models are resampled with the
    SAME indices, preserving the pairing; resampling them independently would
    inflate the interval by adding variance that the paired design removes.
    """
    if list(baseline.uids) != list(proposed.uids):
        raise ValueError(
            "difference_in_differences requires aligned predictions: the two "
            "models must have been evaluated on the same samples in the same "
            "order. Got differing uid sequences."
        )

    cong = np.array([c if c is not None else "unknown" for c in baseline.congruence])
    is_cong = cong == "congruent"
    idx_c = np.flatnonzero(is_cong)
    idx_i = np.flatnonzero(~is_cong & (cong != "unknown"))

    if len(idx_c) == 0 or len(idx_i) == 0:
        raise ValueError(
            "difference_in_differences needs both strata populated; this corpus "
            "does not provide per-modality labels, so the analysis is unavailable."
        )

    def did(sel_c: np.ndarray, sel_i: np.ndarray) -> tuple[float, float, float]:
        gc = (_macro_f1(baseline.y_true[sel_c], proposed.y_pred[sel_c])
              - _macro_f1(baseline.y_true[sel_c], baseline.y_pred[sel_c]))
        gi = (_macro_f1(baseline.y_true[sel_i], proposed.y_pred[sel_i])
              - _macro_f1(baseline.y_true[sel_i], baseline.y_pred[sel_i]))
        return gi - gc, gc, gi

    point, gap_c, gap_i = did(idx_c, idx_i)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sc = rng.choice(idx_c, size=len(idx_c), replace=True)
        si = rng.choice(idx_i, size=len(idx_i), replace=True)
        boot[b] = did(sc, si)[0]

    finite = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite)
              else (float("nan"), float("nan")))

    return {
        "did": point,
        "gap_congruent": gap_c,
        "gap_incongruent": gap_i,
        "ci95": (float(lo), float(hi)),
        "n_congruent": int(len(idx_c)),
        "n_incongruent": int(len(idx_i)),
        "n_boot": int(len(finite)),
        # Not a p-value: the fraction of replicates on the wrong side of zero.
        "frac_le_zero": float((finite <= 0).mean()) if len(finite) else float("nan"),
    }


def sarcasm_slice(p: Predictions, n_labels: int) -> dict[str, dict]:
    """Metrics on the corpus's own sarcasm flag, where one exists.

    Reported alongside congruence because the two are related but not the same
    thing: in Ar-MuSA 13.3% of samples are flagged sarcastic and 66.9% of those
    are also incongruent, so sarcasm is a subset-ish but not a synonym. Keeping
    them separate avoids claiming a sarcasm result on the strength of a
    congruence effect.
    """
    if not p.sarcasm or all(s is None for s in p.sarcasm):
        return {}
    s = np.array([-1 if x is None else int(x) for x in p.sarcasm])
    out = {}
    for name, m in (("sarcastic", s == 1), ("not_sarcastic", s == 0)):
        if m.any():
            out[name] = core_metrics(p.y_true[m], p.y_pred[m], n_labels)
    return out


def format_report(results: dict[str, dict], did: dict | None = None) -> str:
    """Plain-text table. Deliberately terse — it goes straight into the report."""
    rows = ["", f"{'model / stratum':28s} {'n':>6s} {'macro-F1':>9s} {'wt-F1':>8s} {'acc':>7s}",
            "-" * 62]
    for model, strata in results.items():
        for name in ("overall", "congruent", "partial", "conflicting", "incongruent"):
            if name not in strata:
                continue
            m = strata[name]
            rows.append(f"{model + ' / ' + name:28s} {m['n']:6d} "
                        f"{m['macro_f1']:9.4f} {m['weighted_f1']:8.4f} {m['accuracy']:7.4f}")
        rows.append("")

    if did:
        lo, hi = did["ci95"]
        rows += [
            "difference-in-differences (macro-F1)",
            "-" * 62,
            f"  gap on congruent samples   (n={did['n_congruent']:5d}) : {did['gap_congruent']:+.4f}",
            f"  gap on incongruent samples (n={did['n_incongruent']:5d}) : {did['gap_incongruent']:+.4f}",
            f"  DiD                                      : {did['did']:+.4f}",
            f"  bootstrap 95% CI                         : [{lo:+.4f}, {hi:+.4f}]",
            f"  replicates <= 0                          : {did['frac_le_zero']:.1%}",
            "",
        ]
    return "\n".join(rows)
