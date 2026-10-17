"""Milestone 4: do the uncertainty methods actually detect errors?

Usage:
    python -m src.evaluate_detection

Reads the 54 predictive distributions saved by Milestone 3 and scores each one
on the three questions that matter:

    discrimination   AUROC for misclassification detection
    calibration      expected and maximum calibration error
    usefulness       risk-coverage curve, AURC, accuracy at fixed coverage

Nothing is re-run through the network; every number comes from saved
probabilities, so all methods are scored on byte-identical inputs.

Two things to keep in mind when reading the output:

* **Each method is scored against its own errors.** Temperature scaling never
  changes a prediction, so it shares softmax's error set exactly. MC Dropout
  does change predictions, so its error set differs. A method can therefore
  post a higher AUROC partly because it is ranking a different, easier set of
  mistakes.
* **Results are reported on the 4,680 held-out test images** that were not used
  to choose the severity levels, with full-test numbers alongside as a check.

Outputs:
    outputs/metrics/detection_evaluation.json
    outputs/figures/detection_auroc.png
    outputs/figures/detection_calibration.png
    outputs/figures/reliability_diagrams.png
    outputs/figures/risk_coverage.png
    outputs/figures/score_comparison.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, degradations, metrics as M
from .calibrate_severity import calibration_indices

METHOD_LABELS = {
    "softmax": "softmax confidence",
    "temperature": "temperature scaling",
    "mc_dropout": "MC Dropout",
}
METHOD_COLORS = {"softmax": "#3b6ea5", "temperature": "#d08b2c", "mc_dropout": "#4f9153"}
SCORE_COLORS = {"max_prob": "#3b6ea5", "entropy": "#d08b2c",
                "margin": "#8d5fa8", "mutual_info": "#4f9153"}
COVERAGES = (0.5, 0.7, 0.8, 0.9, 1.0)
N_BINS = 15


def load(method: str, corruption: str, severity: int):
    path = config.UNCERTAINTY_DIR / f"{method}_{corruption}_s{severity}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python -m src.evaluate_uncertainty")
    with np.load(path) as d:
        return (d["probs"], d["labels"],
                d["mutual_information"] if "mutual_information" in d else None)


def score_condition(probs, labels, mi, keep_bins: bool) -> dict:
    """All detection metrics for one method on one condition."""
    preds = probs.argmax(axis=1)
    correct = preds == labels
    confidence = probs.max(axis=1)

    scores = M.uncertainty_scores(probs, mi)
    ece, mce, bins = M.expected_calibration_error(confidence, correct, N_BINS)

    out = {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "n_errors": int((~correct).sum()),
        "mean_confidence": float(confidence.mean()),
        "ece": ece,
        "mce": mce,
        "auroc": {k: M.misclassification_auroc(v, correct) for k, v in scores.items()},
        "aurc": M.aurc(scores["max_prob"], correct),
        "accuracy_at_coverage": {
            str(c): M.accuracy_at_coverage(scores["max_prob"], correct, c) for c in COVERAGES
        },
    }
    if keep_bins:
        out["reliability_bins"] = bins
    return out


def main() -> None:
    # Held-out split: the test images not used to choose severity levels.
    probs0, labels0, _ = load("softmax", "gaussian_noise", 0)
    n_test = len(labels0)
    heldout = np.ones(n_test, dtype=bool)
    heldout[calibration_indices(n_test)] = False
    print(f"Scoring on {heldout.sum():,} held-out test images "
          f"({(~heldout).sum():,} excluded as severity-calibration data)\n")

    results: list[dict] = []
    for corruption in degradations.CORRUPTIONS:
        print(f"{corruption}:")
        for severity in range(config.NUM_SEVERITIES + 1):
            row = {"corruption": corruption, "severity": severity,
                   "parameter": degradations.severity_label(corruption, severity),
                   "methods": {}}
            # keep reliability bins only where a diagram is drawn
            keep = severity in (0, config.NUM_SEVERITIES)

            for method in config.UNCERTAINTY_METHODS:
                probs, labels, mi = load(method, corruption, severity)
                row["methods"][method] = {
                    "heldout": score_condition(
                        probs[heldout], labels[heldout],
                        mi[heldout] if mi is not None else None, keep),
                    "full_test": score_condition(probs, labels, mi, False),
                }

            m = row["methods"]
            print(f"  s{severity} {row['parameter']:>12s} | AUROC "
                  f"soft {m['softmax']['heldout']['auroc']['max_prob']:.4f} "
                  f"temp {m['temperature']['heldout']['auroc']['max_prob']:.4f} "
                  f"mc {m['mc_dropout']['heldout']['auroc']['max_prob']:.4f} "
                  f"mc-MI {m['mc_dropout']['heldout']['auroc']['mutual_info']:.4f} | ECE "
                  f"soft {m['softmax']['heldout']['ece']:.3f} "
                  f"temp {m['temperature']['heldout']['ece']:.3f} "
                  f"mc {m['mc_dropout']['heldout']['ece']:.3f}")
            results.append(row)
        print()

    # Does temperature scaling move AUROC at all? (prediction made in Milestone 3)
    diffs = [abs(r["methods"]["temperature"]["heldout"]["auroc"]["max_prob"]
                 - r["methods"]["softmax"]["heldout"]["auroc"]["max_prob"]) for r in results]
    summary = {
        "n_heldout": int(heldout.sum()),
        "n_bins": N_BINS,
        "max_auroc_difference_temperature_vs_softmax": float(max(diffs)),
        "mean_auroc_difference_temperature_vs_softmax": float(np.mean(diffs)),
    }

    out = config.METRIC_DIR / "detection_evaluation.json"
    out.write_text(json.dumps({**summary, "results": results}, indent=2))

    plot_auroc(results, config.FIGURE_DIR / "detection_auroc.png")
    plot_calibration(results, config.FIGURE_DIR / "detection_calibration.png")
    plot_reliability(results, config.FIGURE_DIR / "reliability_diagrams.png")
    plot_risk_coverage(heldout, config.FIGURE_DIR / "risk_coverage.png")
    plot_score_comparison(results, config.FIGURE_DIR / "score_comparison.png")

    print("Temperature scaling vs softmax AUROC: "
          f"max difference {summary['max_auroc_difference_temperature_vs_softmax']:.5f}, "
          f"mean {summary['mean_auroc_difference_temperature_vs_softmax']:.5f}")
    print(f"\nMetrics -> {out}")
    print(f"Figures -> {config.FIGURE_DIR}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _rows(results, corruption):
    return sorted((r for r in results if r["corruption"] == corruption),
                  key=lambda r: r["severity"])


def plot_auroc(results, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = _rows(results, corruption)
        sev = [r["severity"] for r in rows]
        # Softmax is drawn thick and temperature dashed on top of it, because
        # the two coincide almost exactly -- that coincidence is the result,
        # and a thin line hidden underneath would read as a missing series.
        styles = {"softmax": dict(lw=4.5, alpha=0.45, marker="o", ms=9, ls="-"),
                  "temperature": dict(lw=1.6, marker="o", ms=4, ls="--"),
                  "mc_dropout": dict(lw=1.8, marker="o", ms=5, ls="-")}
        for method in config.UNCERTAINTY_METHODS:
            y = [r["methods"][method]["heldout"]["auroc"]["max_prob"] for r in rows]
            ax.plot(sev, y, color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], **styles[method])
        mi = [r["methods"]["mc_dropout"]["heldout"]["auroc"]["mutual_info"] for r in rows]
        ax.plot(sev, mi, "^:", lw=1.5, ms=5, color="#7a1f1f", label="MC Dropout (mutual info)")
        ax.axhline(0.5, ls=":", lw=1, color="black", label="useless (0.5)")
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC for detecting errors")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Can the uncertainty score rank wrong predictions above right ones?\nTemperature scaling (dashed) lies on top of softmax (thick): it cannot reorder predictions", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(results, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = _rows(results, corruption)
        sev = [r["severity"] for r in rows]
        for method in config.UNCERTAINTY_METHODS:
            y = [r["methods"][method]["heldout"]["ece"] for r in rows]
            ax.plot(sev, y, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("expected calibration error (lower is better)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Calibration error: does a stated confidence match observed accuracy?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(results, path) -> None:
    """Reliability diagrams on clean data and at the worst noise severity."""
    conds = [("gaussian_noise", 0, "clean"),
             ("gaussian_noise", config.NUM_SEVERITIES, "noise severity 5")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for r_i, (corruption, sev, title) in enumerate(conds):
        row = next(r for r in _rows(results, corruption) if r["severity"] == sev)
        for c_i, method in enumerate(config.UNCERTAINTY_METHODS):
            ax = axes[r_i, c_i]
            d = row["methods"][method]["heldout"]
            bins = d["reliability_bins"]
            centres = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
            accs = [b["accuracy"] if b["accuracy"] is not None else np.nan for b in bins]
            counts = np.array([b["count"] for b in bins], dtype=float)
            width = 1.0 / len(bins)

            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black", label="perfect")
            ax.bar(centres, accs, width=width * 0.9, color=METHOD_COLORS[method],
                   alpha=0.85, edgecolor="white", linewidth=0.5, label="accuracy")
            if counts.sum():
                ax.bar(centres, counts / counts.max() * 0.18, width=width * 0.9,
                       color="grey", alpha=0.35, label="share of predictions")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{METHOD_LABELS[method]}\n{title} "
                         f"(ECE {d['ece']:.3f})", fontsize=9)
            ax.set_xlabel("confidence")
            if c_i == 0:
                ax.set_ylabel("observed accuracy")
            ax.grid(alpha=0.25)
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Reliability diagrams: bars below the diagonal mean overconfident")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(heldout, path) -> None:
    """Risk-coverage curves: what abstention buys you, clean vs degraded."""
    conds = [("gaussian_noise", 0, "clean"),
             ("gaussian_noise", 3, "noise severity 3"),
             ("gaussian_noise", config.NUM_SEVERITIES, "noise severity 5")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (corruption, sev, title) in zip(axes, conds):
        for method in config.UNCERTAINTY_METHODS:
            probs, labels, mi = load(method, corruption, sev)
            probs, labels = probs[heldout], labels[heldout]
            correct = probs.argmax(1) == labels
            cov, risk = M.risk_coverage_curve(-probs.max(1), correct)
            ax.plot(cov, risk, color=METHOD_COLORS[method], lw=1.6,
                    label=f"{METHOD_LABELS[method]} (AURC {M.aurc(-probs.max(1), correct):.3f})")
        ax.set_title(title)
        ax.set_xlabel("coverage (fraction of predictions kept)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("risk (error rate among kept)")
    fig.suptitle("If the model abstains on its least confident cases, how much does risk fall?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_comparison(results, path) -> None:
    """Which uncertainty score detects errors best, within a single method?"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = _rows(results, corruption)
        sev = [r["severity"] for r in rows]
        for score in ("max_prob", "entropy", "margin", "mutual_info"):
            y = [r["methods"]["mc_dropout"]["heldout"]["auroc"].get(score) for r in rows]
            ax.plot(sev, y, "o-", color=SCORE_COLORS[score], label=score.replace("_", " "))
        ax.axhline(0.5, ls=":", lw=1, color="black")
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC for detecting errors")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Uncertainty scores compared within MC Dropout")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
