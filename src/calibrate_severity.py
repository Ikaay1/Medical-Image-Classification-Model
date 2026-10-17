"""Choose severity levels for each corruption by measuring, not guessing.

Usage:
    python -m src.calibrate_severity

The problem this solves: if severity levels are picked as round numbers, the
range can easily be useless -- every level barely touches accuracy, or every
level destroys the image. Either way there is nothing to measure, and the
degradation study says nothing.

Instead this sweeps each corruption's parameter over a wide range on a fixed
subsample of the test set, records accuracy at each value, and then picks the
five parameter values whose accuracies are spaced as evenly as possible between
clean accuracy and a chosen floor. The chosen floor is well above chance (1/9)
on purpose: a model that has collapsed to random guessing is not an interesting
test of whether uncertainty estimates can flag its mistakes.

Outputs:
    outputs/metrics/severity_calibration.json   full sweep + chosen levels
    outputs/figures/severity_calibration.png    accuracy vs parameter value
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from . import config, data, degradations

# Parameter grids to sweep, written in order of INCREASING corruption strength.
# For contrast that means decreasing factor, since a smaller multiplier washes
# the image out more. The first entry of each grid is the no-op (clean) value.
#
# A first coarse pass showed the interesting region is much narrower than
# expected -- this model loses most of its accuracy between sigma 0.01 and 0.05
# of noise -- so these grids are dense where the accuracy curve is steep and
# sparse in the tails.
SWEEP_GRIDS = {
    "gaussian_noise": [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
                       0.04, 0.05, 0.06, 0.08, 0.12, 0.20],
    "gaussian_blur": [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                      1.2, 1.4, 1.6, 2.0, 3.0],
    "contrast": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
                 0.5, 0.45, 0.4, 0.3, 0.2],
}

# Subsample size for the sweep. The full test set is used later for the real
# evaluation; this only needs to be big enough to place the levels.
CALIBRATION_N = 2500

# Lowest accuracy any severity level should target.
ACCURACY_FLOOR = 0.40


def calibration_indices(n_test: int) -> np.ndarray:
    """Indices of the test images used for the severity sweep.

    These images helped choose the severity levels, so they must not also be
    used to report results. evaluate_heldout.py excludes exactly this set.
    """
    rng = np.random.default_rng(config.SEED)
    return rng.choice(n_test, size=min(CALIBRATION_N, n_test), replace=False)


def accuracy_at(model, images_u8: np.ndarray, labels: np.ndarray, name: str, value: float) -> float:
    """Apply one corruption at one parameter value and return accuracy."""
    func, _ = degradations.CORRUPTIONS[name]
    batch = images_u8.astype(np.float32) / 255.0
    corrupted = func(batch, value, seed=config.SEED)
    logits = model.predict(corrupted, batch_size=config.BATCH_SIZE, verbose=0)
    return float((logits.argmax(axis=1) == labels).mean())


def pick_levels(values: list[float], accs: list[float], clean_acc: float) -> list[float]:
    """Choose five parameter values with roughly evenly spaced accuracies.

    `values` must be ordered by increasing corruption strength, with the no-op
    (clean) value first.

    Targets are equally spaced between clean accuracy and ACCURACY_FLOOR. For
    each target, the FIRST value whose accuracy has fallen to or below that
    target is selected, walking forward through the grid and never stepping
    backward.

    Choosing the first crossing rather than the nearest accuracy matters,
    because these accuracy curves are not monotonic. Under heavy contrast
    reduction accuracy falls below chance and then rises again, so "nearest
    accuracy to the target" can select an absurdly strong corruption that
    happens to score the same as a mild one. Walking forward also guarantees
    the five levels are strictly increasing in strength and never reuse the
    clean value.
    """
    targets = np.linspace(clean_acc, ACCURACY_FLOOR, config.NUM_SEVERITIES + 1)[1:]

    chosen: list[float] = []
    cursor = 1  # index 0 is the no-op value; a severity level must corrupt.
    for target in targets:
        while cursor < len(values) - 1 and accs[cursor] > target:
            cursor += 1
        chosen.append(values[cursor])
        cursor = min(cursor + 1, len(values) - 1)

    if len(set(chosen)) != len(chosen):
        raise RuntimeError(
            f"could not find {config.NUM_SEVERITIES} distinct levels from grid {values}; "
            "widen or densify the sweep grid"
        )
    return chosen


def main() -> None:
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {config.BEST_MODEL_PATH}. Run: python -m src.train")

    model = keras.models.load_model(config.BEST_MODEL_PATH)
    images, labels = data.load_split("test")

    idx = calibration_indices(len(labels))
    images, labels = images[idx], labels[idx]
    print(f"Calibrating on {len(labels):,} test images (floor = {ACCURACY_FLOOR:.2f})\n")

    results: dict[str, dict] = {}
    for name, grid in SWEEP_GRIDS.items():
        accs = []
        print(f"{name}:")
        for value in grid:
            acc = accuracy_at(model, images, labels, name, value)
            accs.append(acc)
            print(f"    {value:>6.2f} -> accuracy {acc:.4f}")

        clean_acc = accs[0]
        chosen = pick_levels(grid, accs, clean_acc)
        chosen_accs = [accs[grid.index(v)] for v in chosen]

        results[name] = {
            "grid": grid,
            "accuracies": accs,
            "clean_accuracy": clean_acc,
            "chosen_levels": chosen,
            "chosen_accuracies": chosen_accs,
        }
        print(f"  chosen levels : {chosen}")
        print(f"  -> accuracies : {[round(a, 4) for a in chosen_accs]}\n")

    out = config.METRIC_DIR / "severity_calibration.json"
    out.write_text(
        json.dumps(
            {
                "calibration_n": int(len(labels)),
                "accuracy_floor": ACCURACY_FLOOR,
                "num_severities": config.NUM_SEVERITIES,
                "results": results,
            },
            indent=2,
        )
    )

    # ---- figure ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res["grid"], res["accuracies"], "o-", color="#3b6ea5", label="sweep")
        ax.plot(
            res["chosen_levels"],
            res["chosen_accuracies"],
            "D",
            color="#c4452d",
            markersize=9,
            label="chosen severities",
        )
        ax.axhline(res["clean_accuracy"], ls="--", lw=1, color="grey", label="clean")
        ax.axhline(1 / config.NUM_CLASSES, ls=":", lw=1, color="black", label="chance")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel(degradations.CORRUPTIONS[name][1])
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if name == "contrast":
            ax.invert_xaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Severity calibration on {len(labels):,} test images")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "severity_calibration.png", dpi=150)
    plt.close(fig)

    print(f"Calibration -> {out}")
    print(f"Figure      -> {config.FIGURE_DIR / 'severity_calibration.png'}")
    print("\nPaste these into config.SEVERITY_LEVELS:")
    for name, res in results.items():
        print(f'    "{name}": {res["chosen_levels"]},')


if __name__ == "__main__":
    main()
