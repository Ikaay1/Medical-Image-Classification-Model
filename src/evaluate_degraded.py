"""Evaluate the baseline on degraded test images (Milestone 2).

Usage:
    python -m src.evaluate_degraded

Runs the trained classifier over the full test set for every corruption at
every severity, including severity 0 (clean) as the anchor, and records how
accuracy and confidence change as the corruption strengthens.

Outputs:
    outputs/predictions/degraded/logits_<corruption>_s<severity>.npz
    outputs/metrics/degraded_evaluation.json
    outputs/figures/degradation_accuracy.png
    outputs/figures/degradation_confidence_gap.png
    outputs/figures/degradation_examples.png

What this milestone does NOT do: temperature scaling, Monte Carlo Dropout, or
the formal AUROC comparison of uncertainty methods. Those are Milestones 3 and
4. The confidence numbers here are descriptive -- they establish that a problem
exists, which is what the later methods are then asked to solve.

The saved logits are the real deliverable. Every later method reads them, so
each method is scored on byte-identical inputs and nothing has to be recomputed.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tensorflow import keras

from . import config, data, degradations
from .evaluate import softmax

DEGRADED_DIR = config.PREDICTION_DIR / "degraded"


def evaluate_one(
    model: keras.Model,
    images_u8: np.ndarray,
    labels: np.ndarray,
    corruption: str,
    severity: int,
) -> dict:
    """Corrupt the test set once, run the model, and summarise the result."""
    batch = images_u8.astype(np.float32) / 255.0
    corrupted = degradations.apply_corruption(batch, corruption, severity)

    logits = model.predict(corrupted, batch_size=config.BATCH_SIZE, verbose=0)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = preds == labels

    np.savez_compressed(
        DEGRADED_DIR / f"logits_{corruption}_s{severity}.npz",
        logits=logits,
        labels=labels,
    )

    return {
        "corruption": corruption,
        "severity": severity,
        "parameter": degradations.severity_label(corruption, severity),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mean_confidence": float(confidence.mean()),
        # The overconfidence gap: how far mean confidence sits above accuracy.
        # A perfectly calibrated model would score 0 here.
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
        "n_errors": int((~correct).sum()),
        "errors_above_90_confidence": int((confidence[~correct] > 0.90).sum()),
        "errors_above_99_confidence": int((confidence[~correct] > 0.99).sum()),
    }


def plot_accuracy(results: list[dict], path) -> None:
    """Accuracy and mean confidence against severity, one panel per corruption."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = sorted(
            (r for r in results if r["corruption"] == corruption),
            key=lambda r: r["severity"],
        )
        sev = [r["severity"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        conf = [r["mean_confidence"] for r in rows]

        ax.plot(sev, conf, "s--", color="#c4452d", label="mean confidence")
        ax.plot(sev, acc, "o-", color="#3b6ea5", label="accuracy")
        ax.fill_between(sev, acc, conf, color="#c4452d", alpha=0.12)
        ax.axhline(1 / config.NUM_CLASSES, ls=":", lw=1, color="black", label="chance")

        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("accuracy / confidence")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Accuracy collapses faster than confidence: the shaded gap is overconfidence"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence_gap(results: list[dict], path) -> None:
    """Confidence on correct vs incorrect predictions, and the error counts."""
    fig, (ax_conf, ax_err) = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"gaussian_noise": "#3b6ea5", "gaussian_blur": "#4f9153", "contrast": "#c4452d"}

    for corruption in degradations.CORRUPTIONS:
        rows = sorted(
            (r for r in results if r["corruption"] == corruption),
            key=lambda r: r["severity"],
        )
        sev = [r["severity"] for r in rows]
        c = colors[corruption]
        label = corruption.replace("_", " ")

        ax_conf.plot(
            sev, [r["mean_confidence_correct"] for r in rows], "o-", color=c, label=f"{label} (correct)"
        )
        ax_conf.plot(
            sev,
            [r["mean_confidence_incorrect"] for r in rows],
            "o--",
            color=c,
            alpha=0.6,
            label=f"{label} (wrong)",
        )

        # Share of all errors that were made at >99% confidence.
        share = [
            100.0 * r["errors_above_99_confidence"] / r["n_errors"] if r["n_errors"] else 0.0
            for r in rows
        ]
        ax_err.plot(sev, share, "o-", color=c, label=label)

    ax_conf.set_xlabel("severity (0 = clean)")
    ax_conf.set_ylabel("mean confidence")
    ax_conf.set_title("Confidence when right vs when wrong")
    ax_conf.set_xticks(range(config.NUM_SEVERITIES + 1))
    ax_conf.grid(alpha=0.3)
    ax_conf.legend(fontsize=7, ncol=1)

    ax_err.set_xlabel("severity (0 = clean)")
    ax_err.set_ylabel("% of errors made above 99% confidence")
    ax_err.set_title("Confidently wrong predictions")
    ax_err.set_xticks(range(config.NUM_SEVERITIES + 1))
    ax_err.grid(alpha=0.3)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(images_u8: np.ndarray, path, image_index: int = 0) -> None:
    """Show one image at every corruption and severity, so the scale is visible."""
    n_cols = config.NUM_SEVERITIES + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(1.7 * n_cols, 5.6))

    base = images_u8[image_index : image_index + 1].astype(np.float32) / 255.0
    for row, corruption in enumerate(degradations.CORRUPTIONS):
        for sev in range(n_cols):
            ax = axes[row, sev]
            out = degradations.apply_corruption(base, corruption, sev)
            ax.imshow(np.clip(out[0], 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title("clean" if sev == 0 else f"s{sev}", fontsize=9)
            if sev == 0:
                ax.set_ylabel(corruption.replace("_", "\n"), fontsize=8)
            else:
                ax.set_xlabel(
                    degradations.severity_label(corruption, sev).split("=")[1], fontsize=7
                )

    fig.suptitle("One test image at each corruption and severity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {config.BEST_MODEL_PATH}. Run: python -m src.train")

    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(config.BEST_MODEL_PATH)
    images, labels = data.load_split("test")
    print(f"Test set: {len(labels):,} images\n")

    results: list[dict] = []
    for corruption in degradations.CORRUPTIONS:
        print(f"{corruption}:")
        for severity in range(config.NUM_SEVERITIES + 1):
            row = evaluate_one(model, images, labels, corruption, severity)
            results.append(row)
            print(
                f"  s{severity} {row['parameter']:>14s}  "
                f"acc={row['accuracy']:.4f}  conf={row['mean_confidence']:.4f}  "
                f"gap={row['confidence_minus_accuracy']:+.4f}  "
                f"errors={row['n_errors']:5d}  >99%conf={row['errors_above_99_confidence']:4d}"
            )
        print()

    out = config.METRIC_DIR / "degraded_evaluation.json"
    out.write_text(json.dumps({"n_test": int(len(labels)), "results": results}, indent=2))

    plot_accuracy(results, config.FIGURE_DIR / "degradation_accuracy.png")
    plot_confidence_gap(results, config.FIGURE_DIR / "degradation_confidence_gap.png")
    plot_examples(images, config.FIGURE_DIR / "degradation_examples.png")

    print(f"Metrics -> {out}")
    print(f"Logits  -> {DEGRADED_DIR}")
    print(f"Figures -> {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
