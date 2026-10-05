"""Baseline CNN architecture for PathMNIST classification.

Two architectural choices here exist to support later milestones:

1. The network ends in a *linear* layer that outputs raw logits, not a softmax.
   Temperature scaling works by dividing logits by a scalar T before the
   softmax, so the logits have to be reachable. Keeping the softmax out of the
   model also avoids the numerical instability of a log(softmax(x)) loss.

2. Dropout layers are present and active during training. Monte Carlo Dropout
   estimates uncertainty by keeping dropout switched on at prediction time and
   averaging many stochastic forward passes, which only works on a model that
   was actually trained with dropout.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def _conv_block(x: tf.Tensor, filters: int, dropout: float) -> tf.Tensor:
    """Two 3x3 convolutions, then downsample by half, then dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=config.BN_MOMENTUM)(x)
        x = layers.ReLU()(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_model(
    *,
    num_classes: int = config.NUM_CLASSES,
    conv_dropout: float = config.CONV_DROPOUT,
    dense_dropout: float = config.DENSE_DROPOUT,
    name: str = config.BASELINE_MODEL_NAME,
) -> keras.Model:
    """Build the compact VGG-style CNN used as this project's baseline.

    Returns an uncompiled model whose output is logits of shape
    (batch, num_classes).
    """
    inputs = keras.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS),
        name="image",
    )

    x = inputs
    for filters in (32, 64, 128, 256):
        x = _conv_block(x, filters, conv_dropout)

    # Global average pooling instead of Flatten keeps the parameter count low
    # and tends to overfit less on a dataset of this size.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=config.BN_MOMENTUM)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dense_dropout)(x)

    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, logits, name=name)


def compile_model(
    model: keras.Model, learning_rate: float = config.LEARNING_RATE
) -> keras.Model:
    """Attach optimizer, loss and metrics. Loss expects logits."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


if __name__ == "__main__":
    # python -m src.model
    m = compile_model(build_model())
    m.summary()
