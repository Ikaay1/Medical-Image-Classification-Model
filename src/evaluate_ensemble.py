"""Deep ensemble: the fourth uncertainty method (Weeks 10-12).

Usage:
    python -m src.evaluate_ensemble
    python -m src.evaluate_ensemble --allow-partial   # test with members trained so far

Five networks with the same architecture, trained with different seeds (see
config.ENSEMBLE_MEMBERS), are each run on all 18 conditions. Their softmax
outputs are averaged into one predictive distribution:

    p_ensemble = mean_k softmax(logits_k)

and their disagreement is measured exactly as MC Dropout's is:

    MI = H(p_ensemble) - mean_k H(softmax(logits_k))

The difference from MC Dropout is where the diversity comes from. MC Dropout
samples sub-networks of ONE trained model; an ensemble trains separate models
that can settle in genuinely different solutions. The literature (Ovadia et
al., 2019) finds ensembles the most robust method under dataset shift, which is
why this project would be incomplete without one.

Three things are recorded beyond the ensemble itself:

* **Each member on its own.** The spread of a single model's AUROC across five
  seeds is the retraining variability that the bootstrap intervals do not
  cover. If two methods differ by less than that spread, the difference is not
  something a practitioner could rely on.
* **Ensemble size.** Metrics for the first 1, 2, ..., 5 members, to show
  whether five is past the point of diminishing returns.
* **Cost.** Five forward passes per image, against 30 for MC Dropout and one
  for softmax and temperature scaling -- but five times the training.

Outputs:
    outputs/predictions/ensemble/<member>_<corruption>_s<sev>.npz   per-member logits
    outputs/predictions/uncertainty/ensemble_<corruption>_s<sev>.npz
    outputs/metrics/ensemble_evaluation.json
    outputs/figures/ensemble_members.png
    outputs/figures/ensemble_size.png
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

from . import config, data, degradations, metrics as M, uncertainty as U
from .calibrate_severity import calibration_indices
from .evaluate_detection import METHOD_COLORS, METHOD_LABELS, N_BINS

MEMBER_DIR = config.PREDICTION_DIR / "ensemble"
MEMBER_DIR.mkdir(parents=True, exist_ok=True)


def combine(member_logits: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average member softmaxes; return (mean_probs, mutual_information)."""
    probs = np.stack([U.softmax(l) for l in member_logits]).astype(np.float64)
    mean = probs.mean(axis=0)
    mean_entropy = np.mean([U.predictive_entropy(p) for p in probs], axis=0)
    mi = U.predictive_entropy(mean) - mean_entropy
    # Tiny negative values can appear from floating-point cancellation.
    return mean.astype(np.float32), np.maximum(mi, 0.0).astype(np.float32)


def detection(probs: np.ndarray, labels: np.ndarray) -> dict:
    correct = probs.argmax(axis=1) == labels
    confidence = probs.max(axis=1)
    return {
        "accuracy": float(correct.mean()),
        "auroc": M.misclassification_auroc(-confidence, correct),
        "ece": M.expected_calibration_error(confidence, correct, N_BINS)[0],
        "aurc": M.aurc(-confidence, correct),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the deep ensemble.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Run with whichever members exist (for testing). The "
                        "ensemble files written are then NOT the final result.")
    args = p.parse_args()

    members = [m for m in config.ENSEMBLE_MEMBERS
               if (config.MODEL_DIR / f"{m}.keras").exists()]
    missing = [m for m in config.ENSEMBLE_MEMBERS if m not in members]
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing members {missing}. Train them with: "
                         "python -m src.train --seed <seed> --name <member>")
    if len(members) < 2:
        raise SystemExit("An ensemble needs at least two members.")
    print(f"Members ({len(members)}): {members}")
    if missing:
        print(f"WARNING: partial ensemble, missing {missing}\n")

    models = {m: keras.models.load_model(config.MODEL_DIR / f"{m}.keras") for m in members}
    images_u8, labels = data.load_split("test")
    heldout = np.ones(len(labels), dtype=bool)
    heldout[calibration_indices(len(labels))] = False

    results = []
    start = time.time()
    for corruption in degradations.CORRUPTIONS:
        print(f"{corruption}:")
        for severity in range(config.NUM_SEVERITIES + 1):
            batch = images_u8.astype(np.float32) / 255.0
            corrupted = degradations.apply_corruption(batch, corruption, severity)

            member_logits = {}
            for m, model in models.items():
                path = MEMBER_DIR / f"{m}_{corruption}_s{severity}.npz"
                if path.exists():
                    with np.load(path) as f:
                        member_logits[m] = f["logits"]
                else:
                    member_logits[m] = model.predict(corrupted, batch_size=config.BATCH_SIZE, verbose=0)
                    np.savez_compressed(path, logits=member_logits[m], labels=labels)

            # The baseline member must reproduce the logits Milestone 2 saved,
            # otherwise the ensemble is not being compared on the same images.
            saved = config.PREDICTION_DIR / "degraded" / f"logits_{corruption}_s{severity}.npz"
            with np.load(saved) as f:
                if not np.allclose(member_logits[config.BASELINE_MODEL_NAME], f["logits"], atol=1e-3):
                    raise RuntimeError(f"{corruption} s{severity}: baseline logits do not reproduce")

            probs, mi = combine(list(member_logits.values()))
            np.savez_compressed(config.UNCERTAINTY_DIR / f"ensemble_{corruption}_s{severity}.npz",
                                probs=probs, labels=labels, mutual_information=mi)

            h = heldout
            singles = {m: detection(U.softmax(l)[h], labels[h]) for m, l in member_logits.items()}
            by_size = []
            for k in range(1, len(members) + 1):
                pk, _ = combine([member_logits[m] for m in members[:k]])
                by_size.append({"members": k, **detection(pk[h], labels[h])})

            ens = detection(probs[h], labels[h])
            ens["auroc_mutual_info"] = M.misclassification_auroc(mi[h], probs[h].argmax(1) == labels[h])
            ens["mean_mutual_information"] = float(mi[h].mean())

            single_auroc = [s["auroc"] for s in singles.values()]
            single_acc = [s["accuracy"] for s in singles.values()]
            row = {
                "corruption": corruption, "severity": severity,
                "parameter": degradations.severity_label(corruption, severity),
                "ensemble": ens,
                "members": singles,
                "member_spread": {
                    "auroc_min": min(single_auroc), "auroc_max": max(single_auroc),
                    "auroc_std": float(np.std(single_auroc, ddof=1)),
                    "accuracy_min": min(single_acc), "accuracy_max": max(single_acc),
                    "accuracy_std": float(np.std(single_acc, ddof=1)),
                },
                "by_size": by_size,
            }
            results.append(row)
            sp = row["member_spread"]
            print(f"  s{severity} {row['parameter']:>12s} | ensemble acc {ens['accuracy']:.3f} "
                  f"AUROC {ens['auroc']:.4f} ECE {ens['ece']:.3f} | members acc "
                  f"{sp['accuracy_min']:.3f}-{sp['accuracy_max']:.3f} AUROC "
                  f"{sp['auroc_min']:.4f}-{sp['auroc_max']:.4f}")
        print()

    out = config.METRIC_DIR / "ensemble_evaluation.json"
    out.write_text(json.dumps({
        "members": members, "partial": bool(missing),
        "n_heldout": int(heldout.sum()), "results": results}, indent=2))

    plot_members(results, members, config.FIGURE_DIR / "ensemble_members.png")
    plot_size(results, config.FIGURE_DIR / "ensemble_size.png")
    print(f"Done in {(time.time() - start) / 60:.1f} min")
    print(f"Metrics -> {out}")


def _rows(results, corruption):
    return sorted((r for r in results if r["corruption"] == corruption),
                  key=lambda r: r["severity"])


def plot_members(results, members, path) -> None:
    """Each member's AUROC against the ensemble's: retraining variability made visible."""
    fig, axes = plt.subplots(1, len(degradations.CORRUPTIONS), figsize=(4.7 * len(degradations.CORRUPTIONS), 4.2), sharey=True)
    for ax, corruption in zip(axes, degradations.CORRUPTIONS):
        rows = _rows(results, corruption)
        sev = [r["severity"] for r in rows]
        for i, m in enumerate(members):
            ax.plot(sev, [r["members"][m]["auroc"] for r in rows], "-", lw=1,
                    color="grey", alpha=0.6, label="single model, one seed" if i == 0 else None)
        ax.plot(sev, [r["ensemble"]["auroc"] for r in rows], "o-", lw=2,
                color=METHOD_COLORS["ensemble"], label=f"ensemble of {len(members)}")
        ax.axhline(0.5, ls=":", lw=1, color="black")
        ax.set_title(corruption.replace("_", " "))
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(sev)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC for detecting errors")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Error detection by each ensemble member alone (grey) and combined\n"
                 "The grey spread is how much retraining with a new seed moves the result")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_size(results, path) -> None:
    """AUROC and accuracy as members are added, at the worst severity of each corruption."""
    fig, (ax_auc, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"gaussian_noise": "#3b6ea5", "gaussian_blur": "#4f9153", "contrast": "#c4452d",
              "stain_shift": "#8d5fa8"}
    for corruption in degradations.CORRUPTIONS:
        for sev, ls in ((0, ":"), (config.NUM_SEVERITIES, "-")):
            if sev == 0 and corruption != next(iter(degradations.CORRUPTIONS)):
                continue  # severity 0 is the same clean data for every corruption
            row = next(r for r in _rows(results, corruption) if r["severity"] == sev)
            k = [s["members"] for s in row["by_size"]]
            label = "clean" if sev == 0 else f"{corruption.replace('_', ' ')} s{sev}"
            color = "black" if sev == 0 else colors[corruption]
            ax_auc.plot(k, [s["auroc"] for s in row["by_size"]], "o" + ls, color=color, label=label)
            ax_acc.plot(k, [s["accuracy"] for s in row["by_size"]], "o" + ls, color=color, label=label)
    for ax, name in ((ax_auc, "AUROC for detecting errors"), (ax_acc, "accuracy")):
        ax.set_xlabel("ensemble members")
        ax.set_xticks(range(1, len(results[0]["by_size"]) + 1))
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    ax_auc.legend(fontsize=8)
    fig.suptitle("Does adding ensemble members keep helping?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
