"""Image degradations applied to the test set only (Milestone 2).

The project asks how a classifier behaves on images that differ from the ones it
was trained on. These three corruptions produce that shift in a controlled,
measurable way: Gaussian noise, Gaussian blur, and reduced contrast.

Conventions used throughout:

* Every function takes and returns float32 images in [0, 1], shaped
  (N, H, W, C). Degradation happens in float space, and the result is clipped
  back to [0, 1] because a real image cannot be darker than black or brighter
  than white. Clipping is part of the corruption, not an afterthought.
* Severity 0 always means "clean". Every severity sweep therefore starts from
  the undegraded image, which gives each curve an anchor point.
* Noise is random, so every call takes an explicit seed. The same severity and
  the same seed always produce the same corrupted test set, which is what makes
  the comparison across uncertainty methods in later milestones fair: each
  method must be scored on byte-identical inputs.

These corruptions are NEVER applied to training data. See the note in
config.py about why training augmentation deliberately excludes contrast.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config


# --------------------------------------------------------------------------
# Individual corruptions
# --------------------------------------------------------------------------
def gaussian_noise(images: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add pixel-wise Gaussian noise with standard deviation `sigma`.

    Models sensor noise or a low-dose / fast acquisition. `sigma` is expressed
    on the [0, 1] intensity scale, so sigma=0.1 is 10% of the full dynamic
    range.
    """
    if sigma <= 0:
        return images.astype(np.float32, copy=True)

    rng = np.random.default_rng(seed)
    noisy = images + rng.normal(0.0, sigma, size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def gaussian_blur(images: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Blur each image with a Gaussian kernel of width `sigma` pixels.

    Models being out of focus or a lower-resolution scanner. The filter is
    applied across height and width only -- never across the batch axis (which
    would smear separate images together) and never across colour channels
    (which would desaturate rather than blur).

    `seed` is accepted and ignored so that every corruption shares one calling
    convention; blur is deterministic.
    """
    if sigma <= 0:
        return images.astype(np.float32, copy=True)

    blurred = gaussian_filter(
        images.astype(np.float32),
        sigma=(0.0, sigma, sigma, 0.0),
        mode="reflect",
    )
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def contrast(images: np.ndarray, factor: float, seed: int | None = None) -> np.ndarray:
    """Scale each image's contrast toward its own mean intensity.

    x' = (x - mean) * factor + mean

    `factor` < 1 reduces contrast (factor=0 would collapse the image to a flat
    grey patch); `factor` > 1 increases it. Reduction is the direction used
    here, since washed-out slides are the realistic failure and the harder case
    for the classifier.

    The mean is computed per image over all pixels and channels, so the overall
    brightness of each image is preserved while the spread around it shrinks.

    `seed` is accepted and ignored; this corruption is deterministic.
    """
    if factor == 1.0:
        return images.astype(np.float32, copy=True)

    means = images.mean(axis=(1, 2, 3), keepdims=True)
    adjusted = (images - means) * factor + means
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
# Each corruption maps to (function, parameter name). The parameter values for
# severities 1-5 live in config.SEVERITY_LEVELS, so that the calibration of
# "how strong is severity 3" is a configuration question rather than something
# buried in this module.
CORRUPTIONS: dict[str, tuple[Callable[..., np.ndarray], str]] = {
    "gaussian_noise": (gaussian_noise, "sigma"),
    "gaussian_blur": (gaussian_blur, "sigma"),
    "contrast": (contrast, "factor"),
}


def apply_corruption(
    images: np.ndarray, name: str, severity: int, seed: int | None = None
) -> np.ndarray:
    """Apply `name` at `severity` (0 = clean, 1-5 = increasing strength).

    Images must be float32 in [0, 1]. The seed defaults to a value derived from
    the corruption name and severity, so that a given severity of a given
    corruption is always reproducible, while different severities get different
    noise draws.
    """
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; expected one of {list(CORRUPTIONS)}")
    if severity == 0:
        return images.astype(np.float32, copy=True)
    if not 1 <= severity <= config.NUM_SEVERITIES:
        raise ValueError(
            f"severity must be 0..{config.NUM_SEVERITIES}, got {severity}"
        )

    func, param_name = CORRUPTIONS[name]
    value = config.SEVERITY_LEVELS[name][severity - 1]

    if seed is None:
        # Stable per (corruption, severity), different across them.
        seed = config.SEED + 1000 * (1 + sorted(CORRUPTIONS).index(name)) + severity

    return func(images, value, seed=seed)


def severity_label(name: str, severity: int) -> str:
    """Human-readable description, e.g. 'sigma=0.10'. Used for axis labels."""
    if severity == 0:
        return "clean"
    _, param_name = CORRUPTIONS[name]
    return f"{param_name}={config.SEVERITY_LEVELS[name][severity - 1]:g}"


if __name__ == "__main__":
    # python -m src.degradations -- sanity check on a small batch
    from . import data

    images, _ = data.load_split("test")
    batch = images[:64].astype(np.float32) / 255.0

    print(f"clean: min={batch.min():.3f} max={batch.max():.3f} mean={batch.mean():.3f}")
    for name in CORRUPTIONS:
        for sev in range(1, config.NUM_SEVERITIES + 1):
            out = apply_corruption(batch, name, sev)
            assert out.dtype == np.float32 and out.shape == batch.shape
            assert 0.0 <= out.min() and out.max() <= 1.0
            rmse = float(np.sqrt(((out - batch) ** 2).mean()))
            print(
                f"{name:15s} sev {sev} {severity_label(name, sev):>14s} "
                f"mean={out.mean():.3f} std={out.std():.3f} rmse_vs_clean={rmse:.4f}"
            )
