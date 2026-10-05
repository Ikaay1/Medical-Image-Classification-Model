# Progress Report 1 — source material

Working notes to draft from. Every number here is reproducible from
`outputs/metrics/` and `outputs/predictions/`. Write the prose in your own
words; this is raw material, not a draft.

---

## 1. What was accomplished this period

Milestone 1 of five: a working environment, a data pipeline, a trained baseline
classifier, and a full evaluation on clean (undegraded) images. This is the
reference point that every later experiment is measured against. No degradation
or uncertainty methods yet — those are Reports 2 and 3.

Concretely:

- Python/TensorFlow environment built and verified on Apple Silicon
- PathMNIST (64x64) downloaded and validated: 89,996 train / 10,004 val / 7,180 test, 9 classes
- `tf.data` input pipeline with label-preserving augmentation
- A 1,244,009-parameter CNN trained for 40 epochs (~165 min on the M4 GPU)
- Evaluation on clean validation and test splits; logits saved for later reuse

---

## 2. Dataset and why

**PathMNIST** (MedMNIST v2), nine tissue classes from colorectal cancer
histology, at **64x64** resolution.

Points to make:

- Publicly available, no credentials, standard benchmark — results are comparable to published work.
- Multiclass (9 classes), which makes misclassification detection a more
  meaningful problem than a binary task would.
- **64x64 rather than the default 28x28**, chosen because later milestones blur
  the test images. At 28x28 there is too little spatial detail for blur to act
  as a gradual corruption; it would destroy the image almost immediately,
  leaving no useful range to measure across.
- The official **test split comes from a different hospital** than train/val
  (`CRC-VAL-HE-7K`). This turns out to matter a lot — see Section 5.

| Split | Images | Source |
|-------|-------:|--------|
| train | 89,996 | NCT-CRC-HE-100K |
| val | 10,004 | NCT-CRC-HE-100K |
| test | 7,180 | CRC-VAL-HE-7K (different hospital) |

---

## 3. Method

**Architecture.** VGG-style CNN: four blocks of (3x3 conv → BatchNorm → ReLU)
x2 → max-pool → dropout, with filter widths 32/64/128/256, then global average
pooling, a 256-unit dense layer, and a linear output layer. 1,244,009 parameters.

**Training.** Adam, initial LR 1e-3, batch size 128, 40 epochs, seed 42.
Cross-entropy on logits. `ReduceLROnPlateau` (factor 0.5, patience 4) and
`EarlyStopping` (patience 8) on validation loss, with the best checkpoint
restored at the end.

### Three design decisions to justify explicitly

These exist to make the *later* milestones possible, and are worth a paragraph
each because they show the project was planned as a whole:

1. **The network outputs raw logits, not softmax probabilities.** Temperature
   scaling works by dividing logits by a scalar T before applying the softmax,
   so the logits must remain reachable. Building the softmax into the model
   would have made Milestone 3 impossible without modifying the architecture.

2. **Dropout is in the architecture from the start** (0.25 in conv blocks, 0.5
   in the dense layer). Monte Carlo Dropout estimates uncertainty by keeping
   dropout active at prediction time and averaging many stochastic forward
   passes. That only works on a model genuinely *trained* with dropout, so
   retrofitting it later would have meant retraining from scratch.

3. **Augmentation was restricted to flips and 90-degree rotations, with no
   brightness or contrast jitter.** Histology tiles have no canonical
   orientation, so the eight symmetries of the square are all label-preserving.
   Contrast jitter was deliberately excluded: Milestone 2 measures how the model
   behaves when test images are degraded by contrast changes, and training with
   contrast augmentation would have made the model robust to precisely the
   corruption under study. This is a threat to validity that was designed out
   rather than discovered later.

---

## 4. Results on clean images

Best epoch 34 of 40 by validation loss (early stopping never triggered).
Final training accuracy 0.9959.

| Metric | Validation | Test |
|--------|-----------:|-----:|
| Accuracy | 0.9945 | 0.9348 |
| Balanced accuracy | 0.9945 | 0.9115 |
| Macro F1 | 0.9944 | 0.9052 |
| Macro AUROC (one-vs-rest) | 1.0000 | 0.9950 |
| Mean max softmax (confidence) | 0.9964 | 0.9697 |

Figures: `outputs/figures/training_curves.png`,
`outputs/figures/confusion_matrix.png`.

**Report balanced accuracy alongside accuracy.** The test split's class balance
differs from training, so plain accuracy flatters the model; the gap between
0.9348 and 0.9115 is exactly that effect.

### Hardest classes (clean test set)

| Class | Precision | Recall | n |
|-------|----------:|-------:|--:|
| cancer-associated stroma | 0.950 | **0.582** | 421 |
| smooth muscle | 0.829 | 0.829 | 592 |
| debris | **0.644** | 1.000 | 339 |
| normal colon mucosa | 0.976 | 0.896 | 741 |

The errors are **structured, not random**, which is worth stating plainly:

- Cancer-associated stroma is the weakest class: only 58% correct, with 19%
  predicted as debris and 12% as smooth muscle.
- That is *why* debris shows perfect recall (1.000) but poor precision (0.644) —
  the model over-predicts debris, absorbing errors from other classes.
- 6% of normal colon mucosa is misclassified as adenocarcinoma epithelium,
  which is the clinically costly direction to get wrong.

This matters for the project: the later milestones have real, systematic,
clinically meaningful errors to try to detect, rather than uniform noise.

---

## 5. The result that motivates the research question

This is the strongest material in the report. Lead with it.

Compare confidence against accuracy on the two splits:

| Split | Accuracy | Mean confidence | Gap |
|-------|---------:|----------------:|----:|
| Validation (same hospital) | 0.9945 | 0.9964 | +0.002 |
| Test (different hospital) | 0.9348 | 0.9697 | **+0.035** |

On validation, confidence and accuracy agree almost exactly — the model is
**well calibrated in-distribution**. On the test split, accuracy falls by 6
points while confidence falls by less than 3, leaving the model **overconfident
by about 3.5 points on data that is only mildly shifted** — a different
hospital, with no artificial degradation applied at all.

Breaking the 468 test errors down by confidence makes the point sharper:

| Quantity | Value |
|----------|------:|
| Test errors | 468 of 7,180 |
| Mean confidence when **correct** | 0.9832 |
| Mean confidence when **wrong** | 0.7766 |
| Errors made at >0.90 confidence | 192 (41.0% of all errors) |
| Errors made at >0.99 confidence | **96 (20.5% of all errors)** |

Two things to draw out:

1. **Confidence carries real signal.** 0.98 when correct versus 0.78 when wrong
   is a large separation. Softmax confidence is not useless, so it is a fair
   baseline to compare the other methods against rather than a straw man.
2. **But it fails exactly where it matters.** 96 predictions were wrong while
   the model reported over 99% confidence. Those are the cases a clinician would
   have no reason to double-check. This is the phenomenon described in the
   proposal, now measured rather than assumed.

Frame this as the justification for the remaining four reports: the problem is
demonstrably present in the baseline, before any image is degraded. Note that
the *formal* comparison — AUROC for misclassification detection and calibration
error — is Milestone 4; Section 5 is descriptive motivation only.

---

## 6. Problem encountered and how it was diagnosed

Worth a proper subsection. It shows methodological care, and an advisor is
likely to ask about the training curves anyway.

**Symptom.** Training accuracy climbed normally while validation accuracy sat at
chance (1/9 ≈ 0.111) for the first several epochs, with the model predicting a
single class for all 10,004 validation images. Validation accuracy across the
first five epochs: 0.493, 0.667, 0.814, **0.461**, **0.500**.

**Why it looked like a data bug.** A model that scores well on training data and
at chance on validation data normally means mismatched images and labels, or a
corrupted split.

**How it was ruled out.** Evaluating the same model with batch statistics
instead of BatchNorm's stored moving averages gave 0.65 on training data and
0.64 on validation — nearly identical, and both healthy. Only inference mode
collapsed, and it collapsed on the *training* data too. A corrupted validation
split cannot explain a failure that also appears on training data, so the
validation split was exonerated.

**Actual cause.** Keras defaults BatchNormalization momentum to 0.99. The true
activation variance in this network is far below BatchNorm's initial value of
1.0, so the leftover initialization still dominated the stored moving variance
after hundreds of updates, and the error compounded across nine BatchNorm
layers. Measured directly: moving variance 0.042 against a true variance far
smaller.

**Why it was not merely cosmetic.** `ModelCheckpoint` and `EarlyStopping` both
select on validation loss. Unreliable early validation numbers therefore corrupt
model selection, not just the plots.

**Fix and verification.** Setting momentum to 0.9 lets the moving statistics
track within a single epoch. Validation accuracy after two identical epochs rose
from 0.109 to 0.640; after full training the stored moving variance matched the
true activation variance to within 0.6%.

**Also worth one sentence:** the training curves show large validation
oscillation through roughly epoch 16 that settles once `ReduceLROnPlateau`
lowers the learning rate. That is learning-rate driven and normal, not a
residual defect.

---

## 7. Environment notes (brief, or an appendix)

Two version conflicts cost real time and are worth recording as evidence of
reproducibility work:

- TensorFlow publishes no wheels for Python 3.14 (the machine's default).
  Python 3.12 also fails, because TensorFlow 2.18 imports `distutils`, which
  3.12 removed from the standard library. The project runs on **Python 3.11**,
  which still ships it.
- `tensorflow-metal` 1.2.0 is ABI-incompatible with TensorFlow 2.21 and breaks
  `import tensorflow` outright. The working pair is **TensorFlow 2.18 +
  tensorflow-metal 1.2.0**, which gives roughly a **3.3x speedup** (1.8 vs 5.9
  min/epoch).
- GPU acceleration was **checked for numerical correctness before being
  trusted**, which matters in a project about confidence calibration: with
  identical weights, GPU and CPU forward passes agreed to a maximum absolute
  difference of 1.8e-7 with 100% agreement on predicted class, and gradient
  norms matched to six significant digits.

---

## 8. Next steps (Milestone 2)

- Implement the three degradations: Gaussian noise, blur, contrast change, each
  at several severity levels.
- Apply them to the test split only, never during training.
- Measure how accuracy and confidence degrade as severity increases, reusing the
  saved logits pipeline.

Expected difficulty to flag now: choosing severity levels that span from
"barely affects accuracy" to "accuracy near chance" without being arbitrary.
Too narrow a range and there is nothing to measure; too wide and the model fails
so completely that no uncertainty method could help.

---

## Numbers quick-reference

```
Parameters            1,244,009
Training time         165.1 min, 40 epochs, best epoch 34
Seed / batch / LR     42 / 128 / 1e-3 (decayed to 3.125e-5)

Validation  acc 0.9945  bal 0.9945  F1 0.9944  AUROC 1.0000  conf 0.9964
Test        acc 0.9348  bal 0.9115  F1 0.9052  AUROC 0.9950  conf 0.9697

Test errors 468 / 7,180
  mean confidence correct    0.9832
  mean confidence incorrect  0.7766
  errors above 0.90 conf     192  (41.0%)
  errors above 0.99 conf      96  (20.5%)
```
