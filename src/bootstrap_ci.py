"""Bootstrap confidence intervals: which differences between methods are real?

Usage:
    python -m src.bootstrap_ci
    python -m src.bootstrap_ci --resamples 200     # quick trial

Milestone 4 reported single numbers. Many of the differences between methods
are small -- a few thousandths of AUROC -- and a single number cannot say
whether a difference that small would survive a different sample of test
images. This script answers that by resampling.

Method:
    The 4,680 held-out test images are resampled with replacement B times. On
    every resample each metric is recomputed for every method, giving B values
    whose 2.5th and 97.5th percentiles form a 95% confidence interval.

    The *same* resampled indices are used for every method within a resample,
    so the comparison between two methods is paired: each difference is
    computed image-set by image-set, and the interval on that difference is
    much tighter than comparing two separate intervals would suggest. A
    difference is called significant when its 95% interval excludes zero.

What the intervals do and do not cover:
    They measure sampling variability over *test images* -- "would this hold on
    another draw of images from the same distribution?" They do not cover the
    run-to-run randomness of MC Dropout (noise floor ~0.004 accuracy, see the
    Milestone 3 caveat) or the variation from retraining the network with a
    different seed. The deep ensemble's members measure the latter directly.

Output:
    outputs/metrics/bootstrap_ci.json
    outputs/figures/detection_auroc_ci.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, degradations, metrics as M
from .calibrate_severity import calibration_indices
from .evaluate_detection import METHOD_COLORS, METHOD_LABELS, N_BINS, load

METRICS = ("auroc", "ece", "aurc", "accuracy")
# Direction of "better", used only to phrase the summary.
HIGHER_IS_BETTER = {"auroc": True, "ece": False, "aurc": False, "accuracy": True}


def available_methods() -> list[str]:
    """Every method whose predictions are on disk, in config order."""
    return [m for m in config.DETECTION_METHODS
            if (config.UNCERTAINTY_DIR / f"{m}_gaussian_noise_s0.npz").exists()]


def ece_fast(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Same quantity as metrics.expected_calibration_error, vectorised.

    The bootstrap calls this tens of thousands of times, so the per-bin Python
    loop is replaced with bincount. Bin edges are identical.
    """
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, N_BINS - 1)
    count = np.bincount(idx, minlength=N_BINS)
    conf_sum = np.bincount(idx, weights=confidence, minlength=N_BINS)
    acc_sum = np.bincount(idx, weights=correct.astype(np.float64), minlength=N_BINS)
    return float(np.abs(conf_sum - acc_sum).sum() / len(confidence))


def point_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    correct = probs.argmax(axis=1) == labels
    confidence = probs.max(axis=1)
    return {
        "auroc": M.misclassification_auroc(-confidence, correct),
        "ece": ece_fast(confidence, correct),
        "aurc": M.aurc(-confidence, correct),
        "accuracy": float(correct.mean()),
    }


def ci(values: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_condition(per_method: dict, resamples: np.ndarray) -> dict:
    """Intervals for every method and every pair of methods on one condition."""
    methods = list(per_method)
    draws = {m: {k: np.empty(len(resamples)) for k in METRICS} for m in methods}

    for b, idx in enumerate(resamples):
        for m in methods:
            probs, labels = per_method[m]
            for k, v in point_metrics(probs[idx], labels[idx]).items():
                draws[m][k][b] = v

    out = {"methods": {}, "differences": {}}
    for m in methods:
        point = point_metrics(*per_method[m])
        out["methods"][m] = {k: {"value": point[k], "ci95": ci(draws[m][k])}
                             for k in METRICS}

    # Every pair, written as (a - b). Paired because draws share indices.
    for a, b in itertools.combinations(methods, 2):
        pair = {}
        for k in METRICS:
            diff = draws[a][k] - draws[b][k]
            lo, hi = ci(diff)
            point = out["methods"][a][k]["value"] - out["methods"][b][k]["value"]
            significant = bool(lo > 0 or hi < 0)
            if not significant:
                verdict = "not distinguishable"
            elif abs(point) < config.PRACTICAL_DIFFERENCE:
                verdict = "real but negligible"
            else:
                verdict = "real"
            pair[k] = {"value": float(point), "ci95": [lo, hi],
                       "significant": significant, "verdict": verdict}
        out["differences"][f"{a}_minus_{b}"] = pair
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap CIs for the detection metrics.")
    p.add_argument("--resamples", type=int, default=config.BOOTSTRAP_RESAMPLES)
    args = p.parse_args()

    methods = available_methods()
    _, labels0, _ = load("softmax", "gaussian_noise", 0)
    heldout = np.ones(len(labels0), dtype=bool)
    heldout[calibration_indices(len(labels0))] = False
    n = int(heldout.sum())

    # One fixed set of resamples, reused for every condition and method.
    rng = np.random.default_rng(config.SEED)
    resamples = rng.integers(0, n, size=(args.resamples, n))
    print(f"{args.resamples} resamples of {n:,} held-out images | methods: {methods}\n")

    start = time.time()
    results = []
    for corruption in degradations.CORRUPTIONS:
        for severity in range(config.NUM_SEVERITIES + 1):
            per_method = {}
            for m in methods:
                probs, labels, _ = load(m, corruption, severity)
                per_method[m] = (probs[heldout], labels[heldout])
            row = {"corruption": corruption, "severity": severity,
                   **bootstrap_condition(per_method, resamples)}
            results.append(row)

            auroc = row["methods"]
            print(f"{corruption:15s} s{severity}  AUROC " + "  ".join(
                f"{m} {auroc[m]['auroc']['value']:.3f} "
                f"[{auroc[m]['auroc']['ci95'][0]:.3f}, {auroc[m]['auroc']['ci95'][1]:.3f}]"
                for m in methods))

    # Of the shifted conditions, how many show a difference that is both
    # statistically real and large enough to matter -- and in whose favour?
    # Severity 0 is the same clean images under every corruption, so counting
    # it once per corruption would count clean data four times. Shifted
    # conditions are tallied here; clean is reported once, on its own.
    shifted = [r for r in results if r["severity"] > 0]
    summary = {"clean": results[0]["differences"]}
    for pair in results[0]["differences"]:
        a, b = pair.split("_minus_")
        summary[pair] = {}
        for k in METRICS:
            diffs = [r["differences"][pair][k] for r in shifted]
            real = [d for d in diffs if d["verdict"] == "real"]
            a_better = sum((d["value"] > 0) == HIGHER_IS_BETTER[k] for d in real)
            summary[pair][k] = {
                f"{a}_better": a_better,
                f"{b}_better": len(real) - a_better,
                "real_but_negligible": sum(d["verdict"] == "real but negligible" for d in diffs),
                "not_distinguishable": sum(d["verdict"] == "not distinguishable" for d in diffs),
            }

    out = config.METRIC_DIR / "bootstrap_ci.json"
    out.write_text(json.dumps({
        "n_heldout": n, "resamples": args.resamples, "methods": methods,
        "practical_difference": config.PRACTICAL_DIFFERENCE,
        "summary": summary, "results": results}, indent=2))

    print(f"\nPaired differences across {len(shifted)} shifted conditions "
          f"(real = 95% CI excludes 0 and |diff| >= {config.PRACTICAL_DIFFERENCE}):")
    for pair, by_metric in summary.items():
        if pair == "clean":
            continue
        a, b = pair.split("_minus_")
        print(f"  {a} vs {b}")
        for k, s in by_metric.items():
            print(f"    {k:9s} {a} better {s[f'{a}_better']:2d} | {b} better {s[f'{b}_better']:2d} | "
                  f"negligible {s['real_but_negligible']:2d} | indistinguishable {s['not_distinguishable']:2d}")
    print("\nClean data (AUROC):")
    for pair, d in summary["clean"].items():
        print(f"  {pair:35s} {d['auroc']['value']:+.4f}  {d['auroc']['verdict']}")

    plot_auroc_ci(results, methods, config.FIGURE_DIR / "detection_auroc_ci.png")
    print(f"\nDone in {time.time() - start:.0f}s")
    print(f"Metrics -> {out}")


def plot_auroc_ci(results, methods, path) -> None:
    fig, axes = plt.subplots(1, len(degradations.CORRUPTIONS), figsize=(4.7 * len(degradations.CORRUPTIONS), 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = sorted((r for r in results if r["corruption"] == corruption),
                      key=lambda r: r["severity"])
        sev = np.array([r["severity"] for r in rows])
        for m in methods:
            y = [r["methods"][m]["auroc"]["value"] for r in rows]
            lo = [r["methods"][m]["auroc"]["ci95"][0] for r in rows]
            hi = [r["methods"][m]["auroc"]["ci95"][1] for r in rows]
            ax.fill_between(sev, lo, hi, color=METHOD_COLORS[m], alpha=0.15, lw=0)
            ax.plot(sev, y, "o-", ms=4, lw=1.6, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.axhline(0.5, ls=":", lw=1, color="black")
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC for detecting errors")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Error-detection AUROC with 95% bootstrap confidence intervals (held-out images)\n"
                 "Overlapping bands are not a test; see the paired differences in bootstrap_ci.json",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
