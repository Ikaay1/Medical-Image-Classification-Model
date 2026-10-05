"""Evaluate the trained baseline on the clean validation and test splits.

Usage:
    python -m src.evaluate

Outputs:
    outputs/predictions/logits_<split>.npz    raw logits + labels per split
    outputs/metrics/clean_evaluation.json     headline and per-class metrics
    outputs/figures/confusion_matrix.png      normalized confusion matrix

Why the logits are saved:
    Every later milestone -- softmax confidence, temperature scaling, Monte
    Carlo Dropout, AUROC of misclassification detection -- is computed from
    logits. Writing them to disk once means later analysis never has to rerun
    the network, and every method is compared on identical predictions.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tensorflow import keras

from . import config, data


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_logits(model: keras.Model, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Run a deterministic forward pass (dropout off) and return logits."""
    ds = data.make_dataset(images, labels, training=False)
    return model.predict(ds, verbose=0)


def evaluate_split(
    model: keras.Model, split: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score one split and return (metrics, logits, labels)."""
    images, labels = data.load_split(split)
    logits = predict_logits(model, images, labels)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    metrics = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        # Balanced accuracy matters here because the PathMNIST test split is
        # drawn from a different hospital and its class balance differs from
        # the training split.
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "macro_auroc_ovr": float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        ),
        "mean_max_softmax": float(probs.max(axis=1).mean()),
    }

    report = classification_report(
        labels,
        preds,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in config.CLASS_NAMES
    }

    return metrics, logits, labels


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, path) -> None:
    """Save a row-normalized confusion matrix (rows = true class)."""
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ticks = range(config.NUM_CLASSES)
    ax.set_xticks(ticks, config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks, config.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Clean test set, row-normalized confusion matrix")

    for i in ticks:
        for j in ticks:
            value = cm_norm[i, j]
            if value >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model at {config.BEST_MODEL_PATH}. Run: python -m src.train"
        )

    print(f"TensorFlow {tf.__version__}")
    model = keras.models.load_model(config.BEST_MODEL_PATH)

    results = {}
    for split in ("val", "test"):
        metrics, logits, labels = evaluate_split(model, split)
        results[split] = metrics

        np.savez_compressed(
            config.PREDICTION_DIR / f"logits_{split}.npz",
            logits=logits,
            labels=labels,
        )

        print(f"\n--- {split} ({metrics['n_samples']:,} images) ---")
        print(f"accuracy          {metrics['accuracy']:.4f}")
        print(f"balanced accuracy {metrics['balanced_accuracy']:.4f}")
        print(f"macro F1          {metrics['macro_f1']:.4f}")
        print(f"macro AUROC (ovr) {metrics['macro_auroc_ovr']:.4f}")
        print(f"mean max softmax  {metrics['mean_max_softmax']:.4f}")

        if split == "test":
            preds = softmax(logits).argmax(axis=1)
            plot_confusion_matrix(
                labels, preds, config.FIGURE_DIR / "confusion_matrix.png"
            )

    out = config.METRIC_DIR / "clean_evaluation.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\nMetrics     -> {out}")
    print(f"Logits      -> {config.PREDICTION_DIR}")
    print(f"Figure      -> {config.FIGURE_DIR / 'confusion_matrix.png'}")

    print("\nPer-class results on the clean test set:")
    print(f"{'class':40s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    for name, m in results["test"]["per_class"].items():
        print(
            f"{name:40s} {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['f1']:6.3f} {m['support']:6d}"
        )


if __name__ == "__main__":
    main()
