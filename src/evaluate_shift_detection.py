"""Can uncertainty detect that the data has shifted, even where it cannot detect errors?

Usage:
    python -m src.evaluate_shift_detection

Milestones 3 and 4 left a claim on record. Mutual information -- disagreement
between MC Dropout's sampled sub-networks -- was the only quantity that rose
monotonically with severity under every corruption, yet it was no better than
plain confidence at flagging individual wrong predictions. The conclusion drawn
was that it is a *shift detector* rather than an *error detector*. That was an
inference; this script measures it.

The question is now "has the input distribution changed?", not "is this
prediction wrong?". Each score is used to separate clean images from corrupted
ones, at two scales:

* **Per image** -- can a single slide be flagged as coming from shifted data?
* **Per batch** -- average the score over a batch of n images and ask the same
  question of the batch. This is how shift would be monitored in practice: a
  lab sees a run of slides from one scanner or one staining session, not
  isolated images. A score that is noisy per image but moves consistently
  with shift should become far more useful once averaged.

Design choice that matters: the held-out images are split into two disjoint
halves, and clean samples are always drawn from one half and corrupted samples
from the other. Without this, every corrupted image would be compared against
its own clean version, and the test would partly measure "is this the same
picture?" instead of "is this distribution different?".

Output:
    outputs/metrics/shift_detection.json
    outputs/figures/shift_detection.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config, degradations, metrics as M
from .calibrate_severity import calibration_indices
from .evaluate_detection import available_methods, load

BATCH_SIZES = (1, 10, 50)
N_BATCHES = 1000

# (label, method, how to get the per-image score). Higher = "looks shifted".
SCORE_LABELS = {
    "softmax_conf": "softmax: 1 - confidence",
    "softmax_entropy": "softmax: entropy",
    "mc_dropout_mi": "MC Dropout: mutual info",
    "ensemble_mi": "ensemble: mutual info",
}
SCORE_COLORS = {"softmax_conf": "#3b6ea5", "softmax_entropy": "#d08b2c",
                "mc_dropout_mi": "#4f9153", "ensemble_mi": "#8d5fa8"}


def per_image_scores(corruption: str, severity: int, methods: list[str]) -> dict[str, np.ndarray]:
    probs, _, _ = load("softmax", corruption, severity)
    scores = {"softmax_conf": 1.0 - probs.max(axis=1),
              "softmax_entropy": M.predictive_entropy(probs)}
    for method in ("mc_dropout", "ensemble"):
        if method in methods:
            _, _, mi = load(method, corruption, severity)
            scores[f"{method}_mi"] = mi.astype(np.float64)
    return scores


def batch_auroc(clean: np.ndarray, shifted: np.ndarray, n: int, rng) -> float:
    """AUROC for telling batch means of clean scores from batch means of shifted ones."""
    if n == 1:
        a, b = clean, shifted
    else:
        a = clean[rng.integers(0, len(clean), size=(N_BATCHES, n))].mean(axis=1)
        b = shifted[rng.integers(0, len(shifted), size=(N_BATCHES, n))].mean(axis=1)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


def main() -> None:
    methods = available_methods()
    _, labels0, _ = load("softmax", "gaussian_noise", 0)
    heldout = np.ones(len(labels0), dtype=bool)
    heldout[calibration_indices(len(labels0))] = False
    idx = np.flatnonzero(heldout)

    # Disjoint halves: clean samples from one, shifted samples from the other.
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(idx)
    clean_half, shifted_half = perm[: len(perm) // 2], perm[len(perm) // 2:]
    print(f"Methods on disk: {methods}")
    print(f"Clean half {len(clean_half):,} images, shifted half {len(shifted_half):,} images\n")

    clean = per_image_scores("gaussian_noise", 0, methods)
    results = []
    for corruption in degradations.CORRUPTIONS:
        for severity in range(1, config.NUM_SEVERITIES + 1):
            shifted = per_image_scores(corruption, severity, methods)
            row = {"corruption": corruption, "severity": severity, "auroc": {}}
            for name in shifted:
                rng = np.random.default_rng(config.SEED + severity)
                row["auroc"][name] = {
                    str(n): batch_auroc(clean[name][clean_half], shifted[name][shifted_half], n, rng)
                    for n in BATCH_SIZES}
            results.append(row)
            print(f"{corruption:15s} s{severity} | " + "  ".join(
                f"{name} " + "/".join(f"{row['auroc'][name][str(n)]:.3f}" for n in BATCH_SIZES)
                for name in row["auroc"]))

    out = config.METRIC_DIR / "shift_detection.json"
    out.write_text(json.dumps({
        "batch_sizes": list(BATCH_SIZES), "n_batches": N_BATCHES,
        "n_clean_half": int(len(clean_half)), "n_shifted_half": int(len(shifted_half)),
        "results": results}, indent=2))
    plot(results, config.FIGURE_DIR / "shift_detection.png")
    print(f"\n(columns are batch sizes {'/'.join(map(str, BATCH_SIZES))})")
    print(f"Metrics -> {out}")


def plot(results, path) -> None:
    """Rows: batch size. Columns: corruption. Lines: score."""
    corruptions = list(degradations.CORRUPTIONS)
    fig, axes = plt.subplots(len(BATCH_SIZES), len(corruptions),
                             figsize=(4.7 * len(corruptions), 3.3 * len(BATCH_SIZES)),
                             sharey=True, sharex=True)
    for r_i, n in enumerate(BATCH_SIZES):
        for c_i, corruption in enumerate(corruptions):
            ax = axes[r_i, c_i]
            rows = sorted((r for r in results if r["corruption"] == corruption),
                          key=lambda r: r["severity"])
            sev = [r["severity"] for r in rows]
            for name in rows[0]["auroc"]:
                ax.plot(sev, [r["auroc"][name][str(n)] for r in rows], "o-", ms=4,
                        color=SCORE_COLORS[name], label=SCORE_LABELS[name])
            ax.axhline(0.5, ls=":", lw=1, color="black")
            ax.set_ylim(0.4, 1.02)
            ax.set_xticks(sev)
            ax.grid(alpha=0.3)
            if r_i == 0:
                ax.set_title(corruption.replace("_", " "))
            if r_i == len(BATCH_SIZES) - 1:
                ax.set_xlabel("severity")
            if c_i == 0:
                ax.set_ylabel(f"{'single image' if n == 1 else f'batch of {n}'}\nAUROC clean vs shifted")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Shift detection: can the score tell shifted data from clean data?\n"
                 "Top row judges one image at a time; lower rows average the score over a batch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
