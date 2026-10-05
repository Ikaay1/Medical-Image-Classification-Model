"""Central configuration for the uncertainty-estimation project.

Every script reads its settings from here so that experiments stay consistent
across the semester. Later milestones (degradations, temperature scaling,
Monte Carlo Dropout) should add their settings to this file rather than
hard-coding values in scripts.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

for _d in (RAW_DATA_DIR, MODEL_DIR, FIGURE_DIR, METRIC_DIR, PREDICTION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# PathMNIST (MedMNIST v2) at 64x64 resolution: nine tissue types from
# colorectal cancer histology slides.
#
# 64x64 was chosen over the default 28x28 because this project later degrades
# the test images with blur and noise. At 28x28 there is too little spatial
# detail for blur to be a meaningful, gradual corruption.
DATASET_NAME = "pathmnist_64"
DATASET_URL = "https://zenodo.org/records/10519652/files/pathmnist_64.npz"
DATASET_PATH = RAW_DATA_DIR / "pathmnist_64.npz"

IMAGE_SIZE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 9

# Order matters: index i is the meaning of label i.
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3

# Dropout is part of the architecture from the very first milestone because
# Monte Carlo Dropout (a later milestone) needs dropout layers that are already
# trained. Retrofitting dropout later would mean retraining the model.
CONV_DROPOUT = 0.25
DENSE_DROPOUT = 0.50

# BatchNormalization momentum for the moving mean/variance used at inference.
#
# Keras defaults to 0.99, which is too slow here. The true activation variance
# in this network is far below BatchNorm's initial value of 1.0, so at 0.99 the
# leftover initialization still dominates the moving variance after hundreds of
# steps, and the effect compounds across nine BatchNorm layers. The symptom is
# validation accuracy pinned near chance for the first several epochs while
# training accuracy climbs normally.
#
# That matters beyond cosmetics: ModelCheckpoint and EarlyStopping both select
# on val_loss, so unreliable early validation numbers corrupt model selection.
# At 0.9 the moving statistics track the true statistics within one epoch.
BN_MOMENTUM = 0.9

# Early stopping / LR schedule
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# --------------------------------------------------------------------------
# Data augmentation
# --------------------------------------------------------------------------
# IMPORTANT (experimental-validity note):
# Augmentation is deliberately limited to flips and 90-degree rotations, which
# are label-preserving for histology tiles (a tissue patch has no canonical
# orientation).
#
# Brightness/contrast jitter is intentionally NOT used. A later milestone
# measures how the model behaves when test images are degraded by contrast
# changes. Training with contrast augmentation would make the model robust to
# exactly the corruption being studied, which would confound that experiment.
USE_AUGMENTATION = True

# --------------------------------------------------------------------------
# Test-set degradations (Milestone 2)
# --------------------------------------------------------------------------
# Applied to the TEST split only, never during training.
#
# Severity 0 is always the clean image. Severities 1-5 were calibrated
# empirically rather than guessed: a sweep over each parameter measured clean
# test accuracy as the parameter varied, and the five levels were chosen to
# spread the resulting accuracy roughly evenly from "barely affected" down to
# "severely degraded". Picking round numbers instead would have risked a range
# where every level either does nothing or destroys the image, leaving nothing
# to measure. See outputs/metrics/severity_calibration.json.
NUM_SEVERITIES = 5

SEVERITY_LEVELS = {
    # Pixel-wise noise standard deviation, on the [0, 1] intensity scale.
    # Note how small these are: this model loses most of its accuracy between
    # sigma 0.01 and 0.05, so noise levels that look negligible are not.
    "gaussian_noise": [0.02, 0.025, 0.03, 0.035, 0.04],
    # Gaussian kernel width in pixels.
    "gaussian_blur": [0.6, 0.8, 0.9, 1.2, 1.6],
    # Contrast multiplier; smaller means more washed out.
    "contrast": [0.85, 0.8, 0.7, 0.65, 0.6],
}

# --------------------------------------------------------------------------
# Artifact filenames
# --------------------------------------------------------------------------
BASELINE_MODEL_NAME = "baseline_cnn"
BEST_MODEL_PATH = MODEL_DIR / f"{BASELINE_MODEL_NAME}.keras"
HISTORY_PATH = METRIC_DIR / f"{BASELINE_MODEL_NAME}_history.json"

# --------------------------------------------------------------------------
# Uncertainty estimation (Milestone 3)
# --------------------------------------------------------------------------
# Temperature scaling needs labelled data to fit its single parameter T, and
# that data must not be the test set. The validation split is used, but it is
# cut in half first: T is fitted on one half and reported on the other, so the
# quality of the calibration is never measured on the data that produced it.
#
# Caveat worth stating in the write-up: the whole validation split was already
# used for early stopping and checkpoint selection during training, so it is
# not perfectly held out. Fitting T on validation data is nonetheless the
# standard protocol (Guo et al., 2017), and a single scalar parameter has very
# little capacity to overfit.
TEMP_FIT_FRACTION = 0.5
TEMPERATURE_SEARCH_BOUNDS = (0.05, 10.0)
TEMPERATURE_PATH = METRIC_DIR / "temperature.json"

# Monte Carlo Dropout: number of stochastic forward passes averaged per image.
# 30 is the usual range in the literature; the cost is linear in this number
# and MC Dropout is the only method that cannot reuse the saved logits.
MC_DROPOUT_SAMPLES = 30

UNCERTAINTY_DIR = PREDICTION_DIR / "uncertainty"
UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)

# The three methods compared from Milestone 3 onward.
UNCERTAINTY_METHODS = ("softmax", "temperature", "mc_dropout")
