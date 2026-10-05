"""Dataset loading and input pipelines for PathMNIST (64x64).

The raw archive stores images as uint8. They are kept as uint8 in memory and
converted to float32 one batch at a time. Converting the whole training split
to float32 up front would need about 4.4 GB of RAM instead of 1.1 GB.
"""

from __future__ import annotations

import urllib.request
from typing import Tuple

import numpy as np
import tensorflow as tf

from . import config


# --------------------------------------------------------------------------
# Download / load
# --------------------------------------------------------------------------
def download_if_missing() -> None:
    """Fetch the PathMNIST archive from Zenodo if it is not already on disk."""
    if config.DATASET_PATH.exists():
        return

    print(f"Downloading {config.DATASET_NAME} to {config.DATASET_PATH} ...")
    partial = config.DATASET_PATH.with_suffix(".npz.part")
    urllib.request.urlretrieve(config.DATASET_URL, partial)
    partial.rename(config.DATASET_PATH)
    print("Download complete.")


def load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for 'train', 'val' or 'test'.

    Images are uint8 with shape (n, 64, 64, 3).
    Labels are int32 with shape (n,) -- the stored (n, 1) column is flattened.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    download_if_missing()
    with np.load(config.DATASET_PATH) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].ravel().astype(np.int32)

    return images, labels


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _to_float(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Scale uint8 [0, 255] to float32 [0, 1]."""
    return tf.cast(images, tf.float32) / 255.0, labels


def _augment(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply label-preserving flips and 90-degree rotations.

    Histology tiles have no canonical orientation, so the eight symmetries of
    the square all produce valid images of the same tissue class.

    No brightness or contrast jitter here on purpose -- see the note in
    config.py about not training away the corruptions studied later.
    """
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)

    # One random rotation of 0, 90, 180 or 270 degrees per batch.
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    images = tf.image.rot90(images, k=k)
    return images, labels


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    training: bool,
    batch_size: int | None = None,
    shuffle_buffer: int = 10_000,
) -> tf.data.Dataset:
    """Build a batched tf.data pipeline.

    training=True adds shuffling and augmentation. Use training=False for
    validation, test and any later evaluation so that results are deterministic.
    """
    batch_size = batch_size or config.BATCH_SIZE
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=config.SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.map(_to_float, num_parallel_calls=tf.data.AUTOTUNE)

    if training and config.USE_AUGMENTATION:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Count examples per class, keyed by human-readable class name."""
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    return {name: int(n) for name, n in zip(config.CLASS_NAMES, counts)}


if __name__ == "__main__":
    # Quick sanity check: python -m src.data
    for split_name in ("train", "val", "test"):
        x, y = load_split(split_name)
        print(f"{split_name:5s} images={x.shape} labels={y.shape} dtype={x.dtype}")
        for cls, count in class_distribution(y).items():
            print(f"    {cls:40s} {count:6d}")
