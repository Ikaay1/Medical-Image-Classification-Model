# Evaluating Uncertainty Estimation Methods for Detecting Incorrect Predictions in Medical Image Classification

Senior Seminar project — Ikechukwu Mgbemele, Fisk University.

**Research question.** How effectively can uncertainty estimation methods
identify incorrect predictions when a medical image classifier is tested on
degraded images that differ from its training data?

This repository is built up across five progress reports. **Only Milestone 1 is
implemented so far.**

---

## Milestone status

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Environment, data pipeline, baseline classifier, clean-test evaluation | **Done** (results below) |
| 2 | Image degradations (Gaussian noise, blur, contrast) + accuracy under shift | **Done** (results below) |
| 3 | Uncertainty methods: softmax confidence, temperature scaling, MC Dropout | Not started |
| 4 | Misclassification-detection metrics (AUROC, calibration error) | Not started |
| 5 | Final comparison, figures, written analysis | Not started |

---

## Milestone 1 results

Trained 40 epochs (~165 min, Apple M4 GPU). Best epoch 34 by validation loss;
`EarlyStopping` never fired. 1,244,009 parameters.

| Metric | Validation | Test |
|--------|-----------:|-----:|
| Accuracy | 0.9945 | 0.9348 |
| Balanced accuracy | 0.9945 | 0.9115 |
| Macro F1 | 0.9944 | 0.9052 |
| Macro AUROC (one-vs-rest) | 1.0000 | 0.9950 |
| Mean max softmax (confidence) | 0.9964 | 0.9697 |

**The two rows that motivate the whole project.** On validation, mean
confidence (0.9964) and accuracy (0.9945) agree almost exactly — the model is
well calibrated in-distribution. On the test split, which comes from a
different hospital, accuracy falls to 0.9348 while confidence stays at 0.9697.
The model is **overconfident by about 3.5 points** on data that is only mildly
shifted, before this project applies any artificial degradation at all.

Hardest classes on the clean test set:

| Class | Precision | Recall | n |
|-------|----------:|-------:|--:|
| cancer-associated stroma | 0.950 | **0.582** | 421 |
| smooth muscle | 0.829 | 0.829 | 592 |
| debris | **0.644** | 1.000 | 339 |

The errors are structured, not random. Cancer-associated stroma is the weakest
class: 19% of it is predicted as debris and 12% as smooth muscle, which is why
debris shows perfect recall but poor precision — the model over-predicts it.
Separately, 6% of normal colon mucosa is misread as adenocarcinoma epithelium.
This gives the later milestones real, clinically meaningful errors to try to
detect, rather than uniform noise.

---

## Milestone 2 results — degraded test images

Three corruptions, five severities each, applied to the **test split only**.
Severity 0 is the clean image and reproduces the Milestone 1 logits exactly.

### Severity levels were calibrated, not guessed

`src/calibrate_severity.py` sweeps each corruption's parameter over a wide
range, measures accuracy at every value, and picks the five levels whose
accuracies are spaced evenly between clean accuracy and a floor of 0.40. The
floor sits well above chance (1/9 = 0.111) on purpose: a model that has
collapsed to random guessing is not an interesting test of whether uncertainty
estimates can flag its mistakes.

This mattered more than expected. The usable range is far narrower than round
numbers would suggest — **this model loses most of its accuracy between sigma
0.01 and 0.05 of noise**. An initial guess of `[0.02, 0.05, 0.09, 0.14, 0.20]`
would have put severities 3, 4 and 5 all at or below chance, producing three
identical, useless data points.

The sweep also found that **accuracy is not monotonic** in corruption strength.
Under heavy contrast reduction accuracy falls to 0.0036 — far *below* chance,
meaning the model is not guessing randomly but confidently predicting one wrong
class — and then rises again as the image collapses to flat grey. Selecting
levels by "nearest accuracy to target" therefore picked absurd values, including
the identity transform. The picker instead walks forward through the grid and
takes the first value to cross each target, which guarantees levels that are
strictly increasing in strength.

| Corruption | Severities 1-5 |
|------------|----------------|
| Gaussian noise (sigma) | 0.02, 0.025, 0.03, 0.035, 0.04 |
| Gaussian blur (sigma, px) | 0.6, 0.8, 0.9, 1.2, 1.6 |
| Contrast (factor) | 0.85, 0.8, 0.7, 0.65, 0.6 |

### Accuracy and confidence under degradation

| Corruption | Metric | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian noise | accuracy | 0.935 | 0.819 | 0.713 | 0.561 | 0.416 | 0.324 |
| | confidence | 0.970 | 0.933 | 0.903 | 0.878 | 0.870 | 0.878 |
| | **gap** | +0.035 | +0.115 | +0.191 | +0.317 | +0.454 | **+0.554** |
| Gaussian blur | accuracy | 0.935 | 0.818 | 0.691 | 0.618 | 0.499 | 0.374 |
| | confidence | 0.970 | 0.930 | 0.893 | 0.881 | 0.854 | 0.857 |
| | **gap** | +0.035 | +0.112 | +0.202 | +0.264 | +0.355 | **+0.483** |
| Contrast | accuracy | 0.935 | 0.770 | 0.705 | 0.578 | 0.437 | 0.328 |
| | confidence | 0.970 | 0.928 | 0.912 | 0.856 | 0.814 | 0.819 |
| | **gap** | +0.035 | +0.158 | +0.207 | +0.278 | +0.376 | **+0.490** |

Figures: `degradation_accuracy.png`, `degradation_confidence_gap.png`,
`degradation_examples.png`, `severity_calibration.png`.

### What the numbers show

**Confidence does not track accuracy.** Under noise, accuracy falls 61 points
(0.935 to 0.324) while mean confidence falls only 9 points (0.970 to 0.878).
The overconfidence gap grows from +0.035 on clean data to **+0.554** — the model
reports 88% confidence while being right 32% of the time.

**Confidence can rise while accuracy falls.** From noise severity 4 to 5,
accuracy drops from 0.416 to 0.324 while mean confidence *increases* from 0.870
to 0.878. Confidence is not merely a lagging indicator of accuracy; the two can
move in opposite directions.

**Confidence becomes less discriminative.** On clean data, mean confidence is
0.983 when correct and 0.777 when wrong. Under corruption the two converge:
confidence on wrong predictions *rises* to roughly 0.81-0.84 while confidence on
correct predictions falls. The signal that separates right from wrong is
weakening exactly where it is needed, which predicts that plain softmax
confidence will lose discriminative power under shift — the hypothesis
Milestone 4 tests formally with AUROC.

**Confidently wrong predictions multiply.** Errors made above 99% confidence go
from 96 on clean data to 1,644 under noise severity 5 — a 17-fold increase, and
34% of all errors at that severity.

**The corruption type matters.** Noise and blur both drive the share of errors
made above 99% confidence *up* to around 34%. Contrast drives it *down* to 8%:
washing the image out makes the model uncertain in a way that noise and blur do
not. Any claim that "the model gets overconfident under shift" is therefore too
coarse — the failure mode depends on which corruption is applied.

**The corruptions are visually mild.** See `degradation_examples.png`. At noise
severity 5 the tissue structure is still clearly readable to a human eye, and
contrast severity 5 is only slightly washed out, yet accuracy is 0.32. These are
not images a person would reject as unusable.

**Pixel-level change does not predict damage.** Measuring how far each severity-5
corruption moves the image (RMSE against the clean image, full test set):

| Corruption | RMSE vs clean | Accuracy | Accuracy lost per unit RMSE |
|------------|--------------:|---------:|----------------------------:|
| Gaussian noise | 0.040 | 0.324 | **15.4** |
| Contrast | 0.064 | 0.328 | 9.5 |
| Gaussian blur | 0.068 | 0.374 | 8.2 |

Noise does roughly the same damage as contrast while changing the pixels only
*two thirds* as much, making it about 1.9x more damaging per unit of pixel
change than blur. The model is disproportionately sensitive to high-frequency
perturbation rather than to overall image distance — which is consistent with a
CNN relying on fine texture cues that noise destroys and blur merely softens.

### What Milestone 3 inherits

The 18 saved logit files are the handover. Two enabling decisions already hold:
the model outputs raw logits, so temperature scaling needs no architectural
change, and it was trained with dropout, so Monte Carlo Dropout runs without
retraining.

Two constraints to respect: temperature must be fitted on a calibration split
held out from validation, never on test; and MC Dropout needs many stochastic
forward passes per image, so unlike the deterministic methods it cannot reuse
the saved logits and must re-run the network across all 18 conditions.

---

## Setup

Requires Homebrew Python 3.11 (see "Why Python 3.11" below).

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools
.venv/bin/python -m pip install -r requirements.txt
```

## Running

```bash
# Inspect the dataset (downloads it on first use, ~1 GB)
.venv/bin/python -m src.data

# Print the model architecture
.venv/bin/python -m src.model

# Train the baseline (~1-2 hours; downloads data if needed)
.venv/bin/python -m src.train

# Quick smoke test instead
.venv/bin/python -m src.train --subset 20000 --epochs 3

# Evaluate on the clean validation and test splits
.venv/bin/python -m src.evaluate

# --- Milestone 2 ---

# Sanity-check the corruption functions on a small batch
.venv/bin/python -m src.degradations

# Re-derive the severity levels (~3 min). Only needed if the model is retrained;
# it prints values to paste into config.SEVERITY_LEVELS.
.venv/bin/python -m src.calibrate_severity

# Evaluate across 3 corruptions x 5 severities + clean (~5 min)
.venv/bin/python -m src.evaluate_degraded
```

Artifacts land in `outputs/` (git-ignored):

```
outputs/models/baseline_cnn.keras          best model by validation loss
outputs/metrics/baseline_cnn_history.json  per-epoch training history
outputs/metrics/clean_evaluation.json      headline + per-class metrics
outputs/predictions/logits_{val,test}.npz  raw logits and labels
outputs/figures/training_curves.png
outputs/figures/confusion_matrix.png

outputs/metrics/severity_calibration.json      severity sweep + chosen levels
outputs/metrics/degraded_evaluation.json       all corruption x severity metrics
outputs/predictions/degraded/*.npz             18 logit files (3 x 6 conditions)
outputs/figures/severity_calibration.png
outputs/figures/degradation_accuracy.png
outputs/figures/degradation_confidence_gap.png
outputs/figures/degradation_examples.png
```

---

## Dataset

**PathMNIST** (MedMNIST v2) at **64x64**: nine tissue classes from colorectal
cancer histology.

| Split | Images |
|-------|--------|
| train | 89,996 |
| val   | 10,004 |
| test  |  7,180 |

Downloaded directly from Zenodo by `src/data.py`. The official `medmnist` pip
package is deliberately **not** a dependency — it pulls in PyTorch and
torchvision (~2.5 GB) that this TensorFlow project has no use for. The archive
is a plain `.npz`.

Two things worth knowing about this dataset:

- **The test split comes from a different hospital** than train/val
  (`CRC-VAL-HE-7K`). Its class balance differs noticeably, so there is already a
  mild distribution shift before this project adds any artificial degradation.
  Expect test accuracy below validation accuracy; that gap is a property of the
  benchmark, not a bug.
- **64x64 was chosen over the default 28x28** because later milestones degrade
  test images with blur. At 28x28 there is too little spatial detail for blur to
  act as a gradual corruption.

---

## Design decisions that exist for later milestones

These are deliberate and should not be "cleaned up":

1. **The model outputs raw logits, not softmax probabilities.** Temperature
   scaling divides logits by a scalar before the softmax, so the logits must be
   reachable. See `src/model.py`.

2. **Dropout is in the architecture from day one.** Monte Carlo Dropout needs a
   model that was genuinely trained with dropout; it cannot be bolted on later
   without retraining.

3. **No brightness or contrast augmentation during training.** Only flips and
   90-degree rotations, which are label-preserving for histology tiles. Training
   with contrast jitter would make the model robust to exactly the corruption
   Milestone 2 studies, confounding the entire experiment.

4. **`src/evaluate.py` saves raw logits to disk.** Every later uncertainty
   method is computed from logits. Saving them once means the analysis in
   Milestones 3-4 never reruns the network, and every method is compared on
   byte-identical predictions.

5. **Corruptions take an explicit seed** (`src/degradations.py`). Gaussian noise
   is random, so without a fixed seed each uncertainty method in Milestone 3
   would be scored on a different corrupted test set and the comparison would be
   meaningless. A given corruption at a given severity always produces the same
   images.

---

## Environment notes

Recorded here because both cost real debugging time.

### Why Python 3.11

The Mac's default `python3` is 3.14, which **TensorFlow has no wheels for**.
Python 3.12 also fails: TensorFlow 2.18 imports `distutils`, which 3.12 removed
from the standard library, and the `setuptools` compatibility shim does not
reliably activate at interpreter startup. Python 3.11 still ships `distutils`,
so the whole problem disappears.

### GPU acceleration

`tensorflow-metal` gives roughly a **3.3x speedup** on this model (about 1.8 vs
5.9 minutes per epoch on an M4).

Version pinning matters: `tensorflow-metal` 1.2.0 is ABI-incompatible with
TensorFlow 2.21 and breaks `import tensorflow` outright with a `dlopen` error.
The working pair is **TensorFlow 2.18 + tensorflow-metal 1.2.0**, which is what
`requirements.txt` pins.

Metal was checked for numerical correctness before being trusted, which matters
for a project about confidence calibration: with identical weights, GPU and CPU
forward passes agreed to a max absolute difference of 1.8e-7 with 100% argmax
agreement, and gradient norms matched to six digits.

### BatchNorm momentum (`config.BN_MOMENTUM = 0.9`)

Keras defaults BatchNormalization momentum to 0.99. With that default this
network shows a striking failure mode: **training accuracy climbs normally while
validation accuracy sits at chance (1/9) for several epochs**, with the model
predicting a single class for every validation image.

It is not a data bug. With batch statistics the same model scored 0.65 on train
and 0.64 on validation; only inference-mode BatchNorm collapsed, and it
collapsed on training data too. The cause is that the true activation variance
here is far below BatchNorm's initial value of 1.0, so at momentum 0.99 the
leftover initialization still dominates the moving variance after hundreds of
steps, compounding across nine BatchNorm layers.

This is not merely cosmetic: `ModelCheckpoint` and `EarlyStopping` both select on
`val_loss`, so unreliable early validation numbers corrupt model selection.
Momentum 0.9 makes the moving statistics track within one epoch (measured:
validation accuracy after two epochs rose from 0.109 to 0.640).

---

## Layout

```
src/config.py              all settings, paths, severity levels
src/data.py                download, loading, tf.data pipelines, augmentation
src/model.py               baseline CNN (1.24M parameters)
src/train.py               training loop, callbacks, curves
src/evaluate.py            clean-split metrics, logits, confusion matrix
src/degradations.py        the three corruptions + severity registry
src/calibrate_severity.py  empirical choice of severity levels
src/evaluate_degraded.py   metrics, logits and figures under degradation
```
