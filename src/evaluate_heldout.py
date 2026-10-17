"""Re-score the degradation study on test images not used to choose severities.

Usage:
    python -m src.evaluate_heldout

calibrate_severity.py chose the five severity levels by measuring accuracy on a
2,500-image subsample of the test set. evaluate_degraded.py then reported
results on the full 7,180-image test set, which includes those same 2,500
images. Tuning and reporting on overlapping data can bias the reported numbers
toward the targets the levels were tuned to hit.

This script removes the overlap. It reads the logits already saved by
evaluate_degraded.py and recomputes every metric on the remaining 4,680
held-out images only. The model is not re-run and the severity levels do not
change; only the set of images being scored does.

Output:
    outputs/metrics/heldout_evaluation.json
"""

from __future__ import annotations

import json

import numpy as np

from . import config, degradations
from .calibrate_severity import calibration_indices
from .evaluate import softmax
from .evaluate_degraded import DEGRADED_DIR


def summarise(logits: np.ndarray, labels: np.ndarray) -> dict:
    """The same headline metrics evaluate_degraded.py reports."""
    probs = softmax(logits)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    return {
        "n_images": int(len(labels)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "n_errors": int((~correct).sum()),
        "errors_above_99_confidence": int((confidence[~correct] > 0.99).sum()),
    }


def main() -> None:
    results = []
    heldout = None

    for corruption in degradations.CORRUPTIONS:
        for severity in range(config.NUM_SEVERITIES + 1):
            path = DEGRADED_DIR / f"logits_{corruption}_s{severity}.npz"
            if not path.exists():
                raise SystemExit(f"Missing {path}. Run: python -m src.evaluate_degraded")
            with np.load(path) as f:
                logits, labels = f["logits"], f["labels"]

            if heldout is None:
                heldout = np.ones(len(labels), dtype=bool)
                heldout[calibration_indices(len(labels))] = False

            full = summarise(logits, labels)
            held = summarise(logits[heldout], labels[heldout])
            results.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "parameter": degradations.severity_label(corruption, severity),
                    "full_test": full,
                    "heldout": held,
                }
            )
            print(
                f"{corruption:15s} s{severity}  "
                f"accuracy full {full['accuracy']:.4f}  held-out {held['accuracy']:.4f}  "
                f"gap held-out {held['confidence_minus_accuracy']:+.4f}"
            )

    max_diff = max(abs(r["full_test"]["accuracy"] - r["heldout"]["accuracy"]) for r in results)
    out = config.METRIC_DIR / "heldout_evaluation.json"
    out.write_text(
        json.dumps(
            {
                "n_heldout": int(heldout.sum()),
                "n_excluded_calibration": int((~heldout).sum()),
                "max_accuracy_difference_full_vs_heldout": max_diff,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nLargest accuracy difference, full vs held-out: {max_diff:.4f}")
    print(f"Held-out evaluation -> {out}")


if __name__ == "__main__":
    main()
