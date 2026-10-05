"""Run all three uncertainty methods over every degraded condition (Milestone 3).

Usage:
    python -m src.evaluate_uncertainty
    python -m src.evaluate_uncertainty --mc-samples 10   # faster trial run

For each of the 18 conditions (3 corruptions x 5 severities, plus clean) this
produces a predictive distribution from each method and records how confident
each one is, how that confidence relates to accuracy, and how well confidence
separates correct from incorrect predictions.

Softmax and temperature scaling are deterministic functions of the logits
already saved by Milestone 2, so they are recomputed from disk rather than by
re-running the network. Monte Carlo Dropout cannot reuse them -- it needs many
stochastic passes -- so it regenerates the corrupted images and runs the model
`--mc-samples` times per condition.

An oracle temperature is also fitted separately on each condition. That is NOT
a deployable method, because it requires labelled data from the corruption you
are about to face. It is reported as an upper bound: the best temperature
scaling could possibly do if the shift were known in advance.

Outputs:
    outputs/predictions/uncertainty/<method>_<corruption>_s<sev>.npz
    outputs/metrics/uncertainty_evaluation.json
    outputs/figures/uncertainty_confidence.png
    outputs/figures/uncertainty_separation.png
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from . import config, data, degradations, uncertainty as U

METHOD_LABELS = {
    "softmax": "softmax confidence",
    "temperature": "temperature scaling",
    "mc_dropout": "MC Dropout",
}
METHOD_COLORS = {"softmax": "#3b6ea5", "temperature": "#d08b2c", "mc_dropout": "#4f9153"}


def summarise(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Descriptive statistics for one method on one condition."""
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = preds == labels
    entropy = U.predictive_entropy(probs)

    conf_ok = float(confidence[correct].mean()) if correct.any() else float("nan")
    conf_bad = float(confidence[~correct].mean()) if (~correct).any() else float("nan")

    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "mean_confidence_correct": conf_ok,
        "mean_confidence_incorrect": conf_bad,
        # How far apart confidence is for right vs wrong answers. This is the
        # quantity that determines whether a confidence threshold can flag
        # errors at all; Milestone 4 turns it into an AUROC.
        "confidence_separation": conf_ok - conf_bad,
        "mean_entropy": float(entropy.mean()),
        "n_errors": int((~correct).sum()),
        "errors_above_99_confidence": int((confidence[~correct] > 0.99).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=config.MC_DROPOUT_SAMPLES)
    args = parser.parse_args()

    if not config.TEMPERATURE_PATH.exists():
        raise SystemExit("Missing temperature. Run: python -m src.fit_temperature")
    temperature = json.loads(config.TEMPERATURE_PATH.read_text())["temperature"]
    print(f"Using T = {temperature:.4f} (fitted on clean validation data)")
    print(f"MC Dropout samples per image: {args.mc_samples}\n")

    model = keras.models.load_model(config.BEST_MODEL_PATH)
    mc_model = U.build_mc_dropout_model(model)
    images_u8, labels = data.load_split("test")

    results: list[dict] = []
    start = time.time()

    for corruption in degradations.CORRUPTIONS:
        print(f"{corruption}:")
        for severity in range(config.NUM_SEVERITIES + 1):
            saved = config.PREDICTION_DIR / "degraded" / f"logits_{corruption}_s{severity}.npz"
            with np.load(saved) as d:
                logits, saved_labels = d["logits"], d["labels"]
            assert np.array_equal(saved_labels, labels), "label order changed"

            # Regenerate the exact images Milestone 2 used, then confirm it by
            # checking the deterministic model reproduces the stored logits.
            batch = images_u8.astype(np.float32) / 255.0
            corrupted = degradations.apply_corruption(batch, corruption, severity)
            check = model.predict(corrupted, batch_size=config.BATCH_SIZE, verbose=0)
            if not np.allclose(check, logits, atol=1e-3):
                raise RuntimeError(
                    f"{corruption} s{severity}: regenerated images do not reproduce "
                    "the stored logits; the corruption is not deterministic"
                )

            per_method = {
                "softmax": U.softmax_probs(logits),
                "temperature": U.temperature_probs(logits, temperature),
            }
            mc_probs, mutual_info = U.mc_dropout_predict(
                mc_model, corrupted, n_samples=args.mc_samples,
                seed=config.SEED + severity,
            )
            per_method["mc_dropout"] = mc_probs

            row = {"corruption": corruption, "severity": severity,
                   "parameter": degradations.severity_label(corruption, severity),
                   "methods": {}}

            for method, probs in per_method.items():
                row["methods"][method] = summarise(probs, labels)
                payload = {"probs": probs, "labels": labels}
                if method == "mc_dropout":
                    payload["mutual_information"] = mutual_info
                np.savez_compressed(
                    config.UNCERTAINTY_DIR / f"{method}_{corruption}_s{severity}.npz", **payload
                )

            row["methods"]["mc_dropout"]["mean_mutual_information"] = float(mutual_info.mean())

            # Oracle upper bound: best possible T for THIS condition. Not
            # deployable -- it needs labels from the corruption being faced.
            t_oracle = U.fit_temperature(logits, labels)
            oracle = summarise(U.temperature_probs(logits, t_oracle), labels)
            oracle["temperature"] = t_oracle
            row["oracle_temperature"] = oracle

            s = row["methods"]
            print(
                f"  s{severity} {row['parameter']:>12s} | "
                f"acc soft {s['softmax']['accuracy']:.3f} mc {s['mc_dropout']['accuracy']:.3f} | "
                f"gap soft {s['softmax']['confidence_minus_accuracy']:+.3f} "
                f"temp {s['temperature']['confidence_minus_accuracy']:+.3f} "
                f"mc {s['mc_dropout']['confidence_minus_accuracy']:+.3f} | "
                f"T* {t_oracle:5.2f} gap {oracle['confidence_minus_accuracy']:+.3f}"
            )
            results.append(row)
        print()

    out = config.METRIC_DIR / "uncertainty_evaluation.json"
    out.write_text(json.dumps({
        "temperature": temperature,
        "mc_samples": args.mc_samples,
        "n_test": int(len(labels)),
        "results": results,
    }, indent=2))

    plot_confidence(results, config.FIGURE_DIR / "uncertainty_confidence.png")
    plot_separation(results, config.FIGURE_DIR / "uncertainty_separation.png")

    print(f"Done in {(time.time()-start)/60:.1f} min")
    print(f"Metrics -> {out}")
    print(f"Scores  -> {config.UNCERTAINTY_DIR}")


def _series(results, corruption, method, field):
    rows = sorted((r for r in results if r["corruption"] == corruption), key=lambda r: r["severity"])
    return [r["severity"] for r in rows], [r["methods"][method][field] for r in rows]


def plot_confidence(results, path) -> None:
    """Mean confidence vs severity for each method, against true accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        sev, acc = _series(results, corruption, "softmax", "accuracy")
        ax.plot(sev, acc, "o-", color="black", lw=2, label="accuracy (truth)")
        for method in config.UNCERTAINTY_METHODS:
            _, conf = _series(results, corruption, method, "mean_confidence")
            ax.plot(sev, conf, "s--", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean confidence")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Every method stays far above the accuracy it should be tracking")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_separation(results, path) -> None:
    """Confidence separation (correct minus incorrect) -- the error-detection signal."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        for method in config.UNCERTAINTY_METHODS:
            sev, sep = _series(results, corruption, method, "confidence_separation")
            ax.plot(sev, sep, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(0, ls=":", lw=1, color="black")
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("confidence: correct minus incorrect")
    axes[0].legend(fontsize=8)
    fig.suptitle("Separation between confidence when right and when wrong (higher is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
