"""Metrics for Milestone 4: does an uncertainty score actually flag errors?

Milestones 2 and 3 measured how *large* confidence is. That is not the same
question as whether confidence is *useful*. Three different things are measured
here, and a method can succeed at one while failing another:

* **Discrimination** -- can the score rank wrong predictions above right ones?
  Measured by AUROC for misclassification detection. Depends only on the
  ordering of scores, not their values.
* **Calibration** -- does a confidence of 0.8 mean right about 80% of the time?
  Measured by expected calibration error. Depends only on the values, and a
  perfectly calibrated model can still be useless at ranking.
* **Usefulness under abstention** -- if the model declines to predict on its
  least confident cases, how much does accuracy recover? Measured by the
  risk-coverage curve. This is the form a clinical workflow would actually use.

Scores are expressed as *uncertainty* throughout: higher means "more likely to
be wrong". Confidence is therefore negated before being used as a score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def misclassification_auroc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for using `uncertainty` to detect incorrect predictions.

    The positive class is "the prediction is wrong". 0.5 is useless (no better
    than flipping a coin); 1.0 would flag every error before any correct
    prediction.

    Returns NaN when every prediction is right or every one is wrong, since
    AUROC is undefined with only one class present.
    """
    errors = (~correct).astype(int)
    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")
    return float(roc_auc_score(errors, uncertainty))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, float, list[dict]]:
    """Equal-width-binned ECE and MCE, plus the per-bin data for a reliability diagram.

    Confidences are placed into `n_bins` equal-width bins over [0, 1]. Within
    each bin the gap between mean confidence and observed accuracy is measured;
    ECE is the average of those gaps weighted by bin population, and MCE is the
    largest single gap.

    ECE is reported far more often than it is scrutinised, so two limitations
    are worth stating. It is sensitive to the number of bins, and it is a
    weighted average, so a model can post a small ECE while being badly wrong
    on a sparsely populated bin -- which is why MCE is returned alongside it.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts values exactly on an edge into the upper bin; clipping
    # keeps confidence == 1.0 inside the last bin rather than overflowing.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    bins: list[dict] = []
    n = len(confidence)

    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
                "count": 0, "accuracy": None, "confidence": None, "gap": None,
            })
            continue

        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(conf - acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({
            "bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
            "count": count, "accuracy": acc, "confidence": conf,
            "gap": float(conf - acc),   # signed: positive means overconfident
        })

    return float(ece), float(mce), bins


# --------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------
def risk_coverage_curve(
    uncertainty: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Error rate as a function of how many predictions are kept.

    Predictions are sorted from most to least certain. At coverage c the model
    answers only on the most certain c fraction and abstains on the rest; risk
    is the error rate among the answered ones.

    Returns (coverage, risk), both length n, with coverage running from 1/n to 1.
    A useful uncertainty score produces a curve that starts near zero risk: the
    predictions it is most sure about really are the right ones.
    """
    order = np.argsort(uncertainty, kind="mergesort")   # most certain first
    errors = (~np.asarray(correct).astype(bool))[order].astype(np.float64)

    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def aurc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better.

    Summarises selective prediction in one number: the average error rate
    across every possible abstention threshold.
    """
    _, risk = risk_coverage_curve(uncertainty, correct)
    return float(risk.mean())


def accuracy_at_coverage(
    uncertainty: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    """Accuracy among the most certain `coverage` fraction of predictions.

    Answers the practical question directly: "if we send the least confident
    20% to a human, how good are the ones we keep?"
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    order = np.argsort(uncertainty, kind="mergesort")
    keep = max(1, int(round(coverage * len(order))))
    return float(np.asarray(correct).astype(bool)[order[:keep]].mean())


# --------------------------------------------------------------------------
# Score constructors
# --------------------------------------------------------------------------
def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, in nats. Higher means less certain."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def uncertainty_scores(probs: np.ndarray, mutual_information: np.ndarray | None = None) -> dict:
    """The candidate uncertainty scores derivable from a predictive distribution.

    * `max_prob`   -- negated top probability. The standard baseline.
    * `entropy`    -- uses the whole distribution, so it separates "two classes
                      at 0.5" from "0.5 and the rest scattered", which
                      max-probability cannot.
    * `margin`     -- negated gap between the top two classes. Targets the
                      specific failure of being torn between two options.
    * `mutual_info`-- MC Dropout only: disagreement between sampled
                      sub-networks, rather than overall ambiguity.
    """
    top2 = np.sort(probs, axis=1)[:, -2:]
    scores = {
        "max_prob": -probs.max(axis=1),
        "entropy": predictive_entropy(probs),
        "margin": -(top2[:, 1] - top2[:, 0]),
    }
    if mutual_information is not None:
        scores["mutual_info"] = np.asarray(mutual_information, dtype=np.float64)
    return scores
