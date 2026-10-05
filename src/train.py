"""Train the baseline PathMNIST classifier on clean (undegraded) images.

Usage:
    python -m src.train
    python -m src.train --epochs 5          # short run, for a smoke test
    python -m src.train --subset 5000       # train on a small subset

Outputs:
    outputs/models/baseline_cnn.keras          best model by validation loss
    outputs/metrics/baseline_cnn_history.json  per-epoch loss and accuracy
    outputs/figures/training_curves.png        loss and accuracy curves
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")  # write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, data
from .model import build_model, compile_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the baseline PathMNIST CNN.")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Train on only the first N training images (for quick tests).",
    )
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def plot_history(history: dict[str, list[float]], path) -> None:
    """Save side-by-side loss and accuracy curves."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)

    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("Baseline CNN on clean PathMNIST (64x64)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    devices = [d.device_type for d in tf.config.list_physical_devices()]
    print(f"TensorFlow {tf.__version__} | devices: {devices}")

    # ---- data --------------------------------------------------------
    x_train, y_train = data.load_split("train")
    x_val, y_val = data.load_split("val")

    if args.subset:
        x_train, y_train = x_train[: args.subset], y_train[: args.subset]

    print(f"train: {len(y_train):,} images | val: {len(y_val):,} images")

    train_ds = data.make_dataset(
        x_train, y_train, training=True, batch_size=args.batch_size
    )
    val_ds = data.make_dataset(
        x_val, y_val, training=False, batch_size=args.batch_size
    )

    # ---- model -------------------------------------------------------
    model = compile_model(build_model(), learning_rate=args.learning_rate)
    print(f"trainable parameters: {model.count_params():,}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    # ---- train -------------------------------------------------------
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,  # one line per epoch; the step-level bar floods log files
    )
    elapsed = time.time() - start

    # ---- persist -----------------------------------------------------
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    record = {
        "history": hist,
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": hist["val_loss"][best_epoch - 1],
        "best_val_accuracy": hist["val_accuracy"][best_epoch - 1],
        "training_seconds": round(elapsed, 1),
        "settings": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "subset": args.subset,
            "augmentation": config.USE_AUGMENTATION,
        },
    }
    config.HISTORY_PATH.write_text(json.dumps(record, indent=2))

    plot_history(hist, config.FIGURE_DIR / "training_curves.png")

    print(
        f"\nDone in {elapsed / 60:.1f} min. "
        f"Best epoch {best_epoch}: "
        f"val_loss={record['best_val_loss']:.4f} "
        f"val_accuracy={record['best_val_accuracy']:.4f}"
    )
    print(f"Model   -> {config.BEST_MODEL_PATH}")
    print(f"History -> {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
