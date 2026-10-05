# Progress Report 2 — source material

Working notes to draft from. Every number is reproducible from
`outputs/metrics/degraded_evaluation.json` and
`outputs/metrics/severity_calibration.json`. Write the prose yourself; this is
raw material, not a draft.

---

## 1. What was accomplished this period

Milestone 2 of five: the degradation study. The baseline from Report 1 was
evaluated on systematically corrupted versions of the test set, and the
relationship between its accuracy and its confidence was measured as the
corruption strengthens.

- Three corruptions implemented: Gaussian noise, Gaussian blur, reduced contrast
- Five severity levels each, **calibrated empirically rather than chosen by hand**
- Full test set (7,180 images) evaluated at all 18 conditions (3 x 5, plus clean)
- Logits saved for every condition, so Milestones 3-4 reuse them unchanged
- Four figures produced

Scope boundary to state explicitly: this report measures *whether* the problem
exists and how it behaves. It does not yet apply temperature scaling or Monte
Carlo Dropout (Milestone 3), nor compute AUROC for misclassification detection
(Milestone 4).

---

## 2. Method

### The corruptions

All three operate on float images in [0, 1] and clip back to that range,
because a real image cannot be darker than black or brighter than white.
Clipping is part of the corruption, not a detail.

| Corruption | Definition | Models |
|---|---|---|
| Gaussian noise | add pixel-wise `N(0, sigma)` | sensor noise, fast/low-dose acquisition |
| Gaussian blur | convolve with Gaussian of width `sigma` px | out-of-focus or lower-resolution scanner |
| Contrast | `x' = (x - mean) * factor + mean`, per image | washed-out or poorly stained slide |

Blur is applied across height and width only — never across the batch axis
(which would smear separate images together) and never across colour channels
(which would desaturate rather than blur).

Two design points worth a sentence each:

- **Corruptions are applied to the test set only, never during training.** This
  is the point of the experiment: the model must meet these conditions for the
  first time at test.
- **Every corruption takes an explicit seed.** Noise is random, so without a
  fixed seed each uncertainty method in Milestone 3 would be scored on a
  different corrupted test set and the comparison would be meaningless. A given
  corruption at a given severity always produces byte-identical images.

### Severity levels were calibrated, not guessed

This is the methodological contribution of Report 2 and deserves its own
subsection.

The procedure (`src/calibrate_severity.py`): sweep each corruption's parameter
over a wide grid on a fixed 2,500-image subsample of the test set, record
accuracy at every value, then choose the five levels whose accuracies fall as
evenly as possible between clean accuracy and a floor of 0.40. The floor sits
well above chance (1/9 = 0.111) deliberately — a model that has collapsed to
random guessing is not an interesting test of whether uncertainty estimates can
flag its mistakes.

**Why this mattered.** Two problems appeared that hand-picked round numbers
would have hidden:

1. *The usable range is far narrower than it looks.* This model loses most of
   its accuracy between sigma 0.01 and 0.05 of noise. An initial guess of
   `[0.02, 0.05, 0.09, 0.14, 0.20]` would have placed severities 3, 4 and 5 all
   at or below chance — three identical, uninformative data points.

2. *Accuracy is not monotonic in corruption strength.* Under heavy contrast
   reduction accuracy falls to **0.0036**, far below chance, which means the
   model is not guessing randomly but confidently predicting one wrong class.
   Push further and accuracy *rises* again as the image collapses toward flat
   grey. Selecting levels by "nearest accuracy to the target" therefore picked
   nonsense, at one point selecting the identity transform (factor = 1.0, no
   corruption at all) as severity 1. The corrected procedure walks forward
   through the grid and takes the first value to cross each target, which
   guarantees levels strictly increasing in strength.

**Chosen levels**, with the accuracy each produced on the calibration subsample:

| Corruption | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| Gaussian noise (sigma) | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 |
| | 0.820 | 0.714 | 0.566 | 0.422 | 0.330 |
| Gaussian blur (sigma px) | 0.6 | 0.8 | 0.9 | 1.2 | 1.6 |
| | 0.815 | 0.677 | 0.610 | 0.494 | 0.368 |
| Contrast (factor) | 0.85 | 0.8 | 0.7 | 0.65 | 0.6 |
| | 0.768 | 0.708 | 0.576 | 0.443 | 0.332 |

The calibration subsample accuracies match the full-test results below to within
about one percentage point, which is a reassuring consistency check.

### Validity check

Severity 0 is the clean image, and its logits were verified to match the
Milestone 1 clean logits **exactly**. This confirms the degradation pipeline
does not accidentally alter images it is meant to leave alone.

---

## 3. Results

Full test set, 7,180 images. "Gap" is mean confidence minus accuracy; a
perfectly calibrated model would score 0.

### Gaussian noise

| Severity | 0 (clean) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| sigma | – | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 |
| Accuracy | 0.9348 | 0.8187 | 0.7127 | 0.5606 | 0.4157 | 0.3235 |
| Balanced accuracy | 0.9115 | 0.7910 | 0.6999 | 0.5783 | 0.4691 | 0.3980 |
| Mean confidence | 0.9697 | 0.9334 | 0.9032 | 0.8778 | 0.8701 | 0.8779 |
| **Gap** | +0.035 | +0.115 | +0.191 | +0.317 | +0.454 | **+0.554** |
| Errors | 468 | 1,302 | 2,063 | 3,155 | 4,195 | 4,857 |
| Errors >99% conf | 96 | 292 | 473 | 755 | 1,183 | **1,644** |

### Gaussian blur

| Severity | 0 (clean) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| sigma (px) | – | 0.6 | 0.8 | 0.9 | 1.2 | 1.6 |
| Accuracy | 0.9348 | 0.8184 | 0.6908 | 0.6178 | 0.4992 | 0.3738 |
| Balanced accuracy | 0.9115 | 0.8053 | 0.7066 | 0.6317 | 0.4851 | 0.3225 |
| Mean confidence | 0.9697 | 0.9302 | 0.8929 | 0.8814 | 0.8541 | 0.8567 |
| **Gap** | +0.035 | +0.112 | +0.202 | +0.264 | +0.355 | **+0.483** |
| Errors | 468 | 1,304 | 2,220 | 2,744 | 3,596 | 4,496 |
| Errors >99% conf | 96 | 323 | 488 | 760 | 1,228 | 1,499 |

### Contrast

| Severity | 0 (clean) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| factor | – | 0.85 | 0.8 | 0.7 | 0.65 | 0.6 |
| Accuracy | 0.9348 | 0.7703 | 0.7053 | 0.5779 | 0.4373 | 0.3284 |
| Balanced accuracy | 0.9115 | 0.7242 | 0.6620 | 0.5196 | 0.3923 | 0.3008 |
| Mean confidence | 0.9697 | 0.9282 | 0.9119 | 0.8556 | 0.8136 | 0.8189 |
| **Gap** | +0.035 | +0.158 | +0.207 | +0.278 | +0.376 | **+0.490** |
| Errors | 468 | 1,649 | 2,116 | 3,031 | 4,040 | 4,822 |
| Errors >99% conf | 96 | 339 | 427 | 654 | 579 | **405** |

---

## 4. Findings — the five things worth arguing

### 4.1 Accuracy collapses; confidence barely moves

Under noise, accuracy falls 61 points (0.935 to 0.324) while mean confidence
falls 9 (0.970 to 0.878). The overconfidence gap grows from +0.035 to **+0.554**:
at severity 5 the model reports 88% confidence while being right 32% of the
time. The same pattern holds for blur (+0.483) and contrast (+0.490).

Figure: `degradation_accuracy.png` — the shaded area between the two curves is
the overconfidence gap, and it visibly widens.

### 4.2 Confidence can rise while accuracy falls

From noise severity 4 to 5, accuracy drops from 0.4157 to 0.3235 while mean
confidence *increases* from 0.8701 to 0.8779. The same reversal appears in blur
(0.8541 to 0.8567) and contrast (0.8136 to 0.8189).

This is the strongest single argument in the report. Confidence is not merely a
lagging or noisy indicator of accuracy — the two can move in opposite
directions. Any workflow that treats a high softmax score as evidence of
reliability has no defence against this.

### 4.3 Confidence becomes less discriminative

The quantity that matters for detecting errors is not absolute confidence but
the *separation* between confidence on correct and on incorrect predictions:

| Separation (conf correct − conf wrong) | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian noise | 0.207 | 0.150 | 0.131 | 0.099 | 0.093 | 0.090 |
| Gaussian blur | 0.207 | 0.140 | 0.122 | 0.106 | 0.084 | 0.086 |
| Contrast | 0.207 | 0.117 | 0.107 | 0.102 | 0.066 | **0.026** |

On clean data the model is meaningfully more confident when it is right (0.983)
than when it is wrong (0.777). Under corruption the two converge, because
confidence on *wrong* predictions actually rises while confidence on correct
ones falls. Under contrast severity 5 the separation is 0.026 — almost nothing.

State the consequence as a prediction: plain softmax confidence should lose most
of its ability to flag errors under shift, and should fail worst under contrast.
Milestone 4 tests this formally with AUROC. Setting up a falsifiable prediction
in Report 2 and testing it in Report 4 is good structure.

Figure: `degradation_confidence_gap.png`, left panel.

### 4.4 Confidently wrong predictions multiply — but not for every corruption

Errors made above 99% confidence rise from 96 on clean data to **1,644** under
noise severity 5, a 17-fold increase and 34% of all errors at that severity.
Blur behaves the same way (1,499 errors, 33%).

Contrast does the opposite. Its share of errors above 99% confidence *falls*,
from 20.5% on clean data to **8.4%** at severity 5. Washing the image out makes
the model broadly uncertain in a way that noise and blur do not.

So the coarse claim "the model becomes overconfident under distribution shift"
is wrong as stated. The overconfidence *gap* grows under all three corruptions,
but the mechanism differs: noise and blur produce confidently wrong predictions,
while contrast produces diffusely uncertain ones. This matters for Milestone 3,
because methods that work by softening peaked distributions (temperature
scaling) may help noise and blur far more than contrast.

Figure: `degradation_confidence_gap.png`, right panel.

### 4.5 Pixel-level change does not predict damage

Measuring how far each severity-5 corruption moves the image (RMSE against the
clean image, full test set):

| Corruption | RMSE vs clean | Accuracy | Accuracy lost per unit RMSE |
|---|---:|---:|---:|
| Gaussian noise | 0.040 | 0.324 | **15.4** |
| Contrast | 0.064 | 0.328 | 9.5 |
| Gaussian blur | 0.068 | 0.374 | 8.2 |

Noise does about the same damage as contrast while changing the pixels only two
thirds as much, and is roughly 1.9x more damaging per unit of pixel change than
blur. The model is disproportionately sensitive to high-frequency perturbation
rather than to overall image distance — consistent with a CNN relying on fine
texture cues that noise destroys and blur merely softens.

### 4.6 The corruptions are visually mild

Worth one sentence plus the figure. At noise severity 5 the tissue structure is
still clearly readable to a human eye, and contrast severity 5 is only slightly
washed out — yet accuracy is around 0.32. These are not images a pathologist
would reject as unusable, which is precisely what makes silent failure dangerous.

Figure: `degradation_examples.png`.

---

## 5. Threats to validity worth acknowledging

Raising these yourself is stronger than having them raised for you.

- **Synthetic corruptions are not real-world shift.** Gaussian noise and
  Gaussian blur are idealised. Real scanner artefacts, staining variation and
  compression differ. The counter-argument is that synthetic corruptions are
  controllable and gradable, which is what a severity study requires — and that
  the test split already carries genuine cross-hospital shift as a separate
  check.
- **One architecture, one dataset.** The finding that noise is more damaging
  than blur may not generalise to other models.
- **The contrast direction is a choice.** Only contrast *reduction* was studied,
  since washed-out slides are the realistic failure. Contrast increase was not
  tested.
- **Severity levels are tied to this model.** They were calibrated against this
  classifier's accuracy, so they are not transferable to a different model
  without re-running the calibration. That is the cost of calibrating rather
  than guessing; the benefit is a usable measurement range.

---

## 6. Next steps (Milestone 3)

Implement the three uncertainty methods and produce an uncertainty score for
every one of the 18 conditions:

- softmax confidence (already available from the saved logits)
- temperature scaling
- Monte Carlo Dropout

Two enabling decisions from Report 1 pay off here: the model outputs raw logits,
so temperature scaling needs no architectural change, and it was trained with
dropout, so MC Dropout runs without retraining.

Difficulties to flag now:

- Temperature must be fitted on a **calibration split held out from validation**.
  Fitting it on the test set would leak and invalidate the comparison.
- An open question the data already raises: should temperature be fitted once on
  clean data, or refitted per corruption? Fitting per corruption assumes you
  know the corruption at deployment time, which you generally do not. Fitting
  once on clean data is the honest setting, and the results above suggest it
  will underperform badly at high severity.
- **MC Dropout cannot reuse the saved logits.** It needs many stochastic forward
  passes per image across all 18 conditions, so it is the expensive method.

---

## Figure guide

All four are in `outputs/figures/` at 150 dpi and are ready to drop into the
report. Suggested order and captions:

**Figure 1 — `degradation_examples.png`** *(put this early, it sets up everything)*
> One test image under each corruption at each severity. Even at severity 5 the
> tissue structure remains readable to a human eye, yet classifier accuracy
> falls to roughly 0.32.

Use it to make the "visually mild, catastrophic for the model" point before any
numbers appear. A reader who has seen this figure takes the rest seriously.

**Figure 2 — `severity_calibration.png`** *(supports Section 2)*
> Accuracy against corruption strength on a 2,500-image subsample. Red diamonds
> mark the five chosen severity levels; the dotted line is chance (1/9).

The noise panel argues the methodology on its own: all five levels are packed
into a narrow cliff between sigma 0.02 and 0.04, with a plateau beyond it where
additional corruption changes nothing. Point at that plateau when explaining why
round-number severities would have failed.

**Figure 3 — `degradation_accuracy.png`** *(your main result — Section 4.1)*
> Accuracy and mean confidence against severity. The shaded region is the
> overconfidence gap, which widens from +0.035 on clean images to +0.554 under
> noise at severity 5.

If the report only has room for one figure, use this one.

**Figure 4 — `degradation_confidence_gap.png`** *(Sections 4.3 and 4.4)*
> Left: mean confidence on correct (solid) versus incorrect (dashed)
> predictions; the two converge as severity rises. Right: the share of errors
> made above 99% confidence, which rises for noise and blur but falls for
> contrast.

This is the two-panel figure that carries the most argument per square inch, but
it needs the most explanation — give it a real paragraph, not a passing
reference. The left panel is the evidence for your Milestone 4 prediction; the
right panel is the evidence that corruption type matters.

A note on ordering: Figures 3 and 4 answer "what happened", Figure 2 answers
"how do you know the experiment was set up fairly". Some readers want the
methodology first; if your course expects a Methods section before Results, put
Figure 2 there and lead Results with Figure 3.

---

## Numbers quick-reference

```
Test set                7,180 images; 18 conditions (3 corruptions x 5 severities + clean)
Calibration             2,500-image subsample, accuracy floor 0.40, chance = 0.111
Severity 0 check        logits match Milestone 1 clean logits exactly

Clean                   acc 0.9348   conf 0.9697   gap +0.035   errors 468   >99% conf 96

Severity 5:
  gaussian noise        acc 0.3235   conf 0.8779   gap +0.554   errors 4,857  >99% conf 1,644 (33.8%)
  gaussian blur         acc 0.3738   conf 0.8567   gap +0.483   errors 4,496  >99% conf 1,499 (33.3%)
  contrast              acc 0.3284   conf 0.8189   gap +0.490   errors 4,822  >99% conf   405 ( 8.4%)

Confidence separation (correct − wrong)
  clean 0.207  ->  noise s5 0.090   blur s5 0.086   contrast s5 0.026

Non-monotonic confidence: noise s4 -> s5, accuracy 0.4157 -> 0.3235 while confidence 0.8701 -> 0.8779
Below-chance accuracy:    contrast factor 0.30 -> accuracy 0.0036 (chance = 0.111)
```
