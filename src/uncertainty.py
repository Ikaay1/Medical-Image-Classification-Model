"""The three uncertainty estimation methods compared in this project.

1. Softmax confidence  -- the unmodified network output. The baseline.
2. Temperature scaling -- divide logits by a single scalar T fitted on held-out
                          labelled data. Cannot change which class is predicted,
                          only how confident the prediction is.
3. Monte Carlo Dropout -- keep dropout active at prediction time and average
                          many stochastic forward passes. Can change the
                          prediction, and yields an epistemic uncertainty term.

A note on what each method can and cannot do, because it matters for reading
the results: temperature scaling is a pure recalibration -- accuracy is
mathematically unchanged, because dividing every logit by the same positive
number cannot reorder them. MC Dropout averages over different sub-networks, so
its predictions, and therefore its accuracy, can differ from the deterministic
model.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from tensorflow import keras

from . import config


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable row-wise softmax of `logits / temperature`."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, in nats.

    Higher means the model is spreading its belief across more classes. Unlike
    max-probability this uses the whole distribution, so it distinguishes
    "0.5 / 0.5 between two classes" from "0.5 and the rest scattered".
    """
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# --------------------------------------------------------------------------
# 1. Softmax confidence
# --------------------------------------------------------------------------
def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Plain softmax -- the uncalibrated baseline."""
    return softmax(logits, temperature=1.0)


# --------------------------------------------------------------------------
# 2. Temperature scaling
# --------------------------------------------------------------------------
def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean NLL of the true classes after dividing logits by `temperature`."""
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the T minimising NLL on the given labelled data.

    NLL is the standard objective (Guo et al., 2017): it is minimised by the
    temperature that makes the predicted probabilities match observed accuracy,
    which is exactly what calibration means. Accuracy is not usable as an
    objective here because T cannot change any prediction.

    One scalar over a bounded interval, so a bounded scalar minimiser is both
    sufficient and more robust than gradient descent.
    """
    result = minimize_scalar(
        lambda t: negative_log_likelihood(logits, labels, t),
        bounds=config.TEMPERATURE_SEARCH_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(result.x)


def temperature_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a fitted temperature."""
    return softmax(logits, temperature)


# --------------------------------------------------------------------------
# 3. Monte Carlo Dropout
# --------------------------------------------------------------------------
class MCDropout(keras.layers.Dropout):
    """A Dropout layer that stays active at prediction time.

    Keras switches dropout off whenever `training=False`. Subclassing and
    forcing the flag is the standard way to keep it on, and is preferable to
    calling `model(x, training=True)`: that flag is global, so it would also
    switch BatchNormalization into batch-statistics mode.

    That distinction is not academic in this project. Milestone 1 measured the
    two BatchNorm modes diverging enormously (0.65 versus 0.11 accuracy while
    the moving averages were still settling). Running MC Dropout with
    `training=True` would therefore silently corrupt every uncertainty estimate
    by also changing the normalisation statistics.
    """

    def call(self, inputs, training=None):  # noqa: ARG002 - flag deliberately ignored
        return super().call(inputs, training=True)


def build_mc_dropout_model(model: keras.Model, force_rate: float | None = None) -> keras.Model:
    """Clone `model`, replacing every Dropout with an always-on MCDropout.

    BatchNormalization layers are cloned unchanged, so they keep using their
    stored moving statistics at prediction time. Dropout carries no weights, so
    the weight ordering is identical and can be copied straight across.

    `force_rate` overrides every dropout rate. Setting it to 0.0 turns the
    clone into an exact copy of the deterministic model, which is how the
    correctness of this surgery is verified.
    """

    def clone_fn(layer: keras.layers.Layer) -> keras.layers.Layer:
        if isinstance(layer, keras.layers.Dropout):
            cfg = layer.get_config()
            if force_rate is not None:
                cfg["rate"] = force_rate
            return MCDropout.from_config(cfg)
        return layer.__class__.from_config(layer.get_config())

    clone = keras.models.clone_model(model, clone_function=clone_fn)
    clone.set_weights(model.get_weights())
    return clone


def mc_dropout_predict(
    mc_model: keras.Model,
    images: np.ndarray,
    n_samples: int = config.MC_DROPOUT_SAMPLES,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Average `n_samples` stochastic forward passes.

    Returns (mean_probs, mutual_information).

    Mutual information is the gap between the entropy of the averaged
    prediction and the average entropy of the individual predictions:

        MI = H(mean_p) - mean(H(p_i))

    It isolates *disagreement between the sampled sub-networks*. Total
    predictive entropy conflates two different things -- an image that is
    genuinely ambiguous (every sub-network agrees it is a coin flip) and an
    image the model has no idea about (the sub-networks disagree). Only the
    second is the kind of uncertainty extra training data would reduce, and it
    is the one that should rise on corrupted inputs.
    """
    keras.utils.set_random_seed(seed)

    summed = np.zeros((len(images), config.NUM_CLASSES), dtype=np.float64)
    summed_entropy = np.zeros(len(images), dtype=np.float64)

    for _ in range(n_samples):
        logits = mc_model.predict(images, batch_size=batch_size, verbose=0)
        probs = softmax(logits)
        summed += probs
        summed_entropy += predictive_entropy(probs)

    mean_probs = (summed / n_samples).astype(np.float32)
    mutual_information = predictive_entropy(mean_probs) - (summed_entropy / n_samples)
    # Tiny negative values can appear from floating-point cancellation.
    return mean_probs, np.maximum(mutual_information, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# Validation split used to fit the temperature
# --------------------------------------------------------------------------
def split_validation(n: int, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split validation indices into (fit, eval) halves."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(round(n * config.TEMP_FIT_FRACTION))
    return order[:cut], order[cut:]
