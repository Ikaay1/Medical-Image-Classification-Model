# Midterm Video Script (~4 minutes)

Spoken words are in quotes. `[SCREEN: ...]` is what to show. The assignment asks
you to demonstrate the actual program, not only slides, so there's one live
command in the middle.

**Before recording:** run every command once so the dataset is cached and you know
the timings. Increase your terminal font size. Open the figures in tabs beforehand
so you can switch instantly.

**Pace:** this is about 4 minutes if you speak normally and don't rush. The live
command adds a pause you talk over.

---

## 0:00 – 0:25 — What the project is

`[SCREEN: report title page, or the terminal]`

> "Hi, I'm Ikechukwu Mgbemele. My project evaluates whether uncertainty estimation
> methods can detect incorrect predictions in medical image classification.
>
> A model can do well on clear images and still fail when they have noise, blur, or
> low contrast. The dangerous part isn't that it's wrong. It's that it's wrong *and
> confident*, so nothing tells a doctor to double-check it. I'm testing whether
> uncertainty methods can give that warning. This is milestone three of five."

---

## 0:25 – 1:00 — What I built

`[SCREEN: editor with the file tree visible]`

> "I've finished three of five milestones: a trained classifier, a pipeline that
> degrades the test images in controlled ways, and all three uncertainty methods."

`[SCREEN: src/model.py, scrolled to the last layer]`

> "Two decisions here were made early so the later work would be possible. The model
> outputs raw scores instead of probabilities, because temperature scaling needs
> those. And dropout is in the network from the start, because Monte Carlo Dropout
> only works on a model trained with it. Neither could be added later without
> retraining."

`[SCREEN: src/config.py, at the augmentation comment]`

> "I also deliberately left contrast out of my training augmentation. Training the
> model to handle contrast changes would make it robust to exactly what I'm testing."

---

## 1:00 – 1:45 — Live demonstration

`[SCREEN: terminal. Type and run:]`

```
.venv/bin/python -m src.evaluate
```

> "This loads my trained model and evaluates it on the clean test set."

`[Let it run — talk over the pause:]`

> "The dataset is PathMNIST: nine tissue types from colorectal cancer slides, about
> ninety thousand training images. The test images come from a different hospital
> than the training images, so there's already a small shift before I change
> anything."

`[SCREEN: point at the output]`

> "Validation accuracy is 99.45 percent, test accuracy 93.48. But confidence is
> 96.97 — on a test set where it's only right 93.48 percent of the time. It's
> overconfident by three and a half points, and I haven't degraded anything yet."

---

## 1:45 – 2:50 — Results

`[SCREEN: degradation_examples.png]`

> "These are my altered test images — noise, blur, and reduced contrast, five levels
> each. The important thing is that even the strongest level is still readable. A
> person wouldn't reject these."

`[SCREEN: degradation_accuracy.png]`

> "Here's what the model does on them. Black is accuracy, dashed is confidence.
> Accuracy falls from 93 percent to 32 under noise. Confidence only drops from 97 to
> 88. The shaded gap is the overconfidence, and it keeps growing.
>
> It gets worse. Between the last two noise levels, accuracy drops but confidence
> goes *up*. They can move in opposite directions."

`[SCREEN: uncertainty_confidence.png]`

> "So I applied the three methods. All three lines are still far above the black
> accuracy line. Every method reduces the overconfidence and none fixes it — the
> best still reports 77 percent confidence while being right 37 percent of the time.
> And no method wins everywhere. Monte Carlo Dropout is best under noise, but under
> blur and contrast it depends on the severity."

---

## 2:50 – 3:20 — A problem I solved

`[SCREEN: training_curves.png]`

> "One problem worth showing. Early in training, training accuracy climbed normally
> but validation accuracy sat at chance — one class predicted for all ten thousand
> validation images. It looked like a data bug.
>
> I ruled that out by evaluating the same model a different way and getting 0.65 on
> training and 0.64 on validation. The failure appeared on training data too, so the
> validation split wasn't broken. It was Batch Normalization's stored statistics.
> One setting fixed it, and validation accuracy went from 10.9 to 64 percent."

---

## 3:20 – 3:50 — What's next

`[SCREEN: file tree, or your report's next-steps section]`

> "Milestone four is the measurement that answers my question: AUROC for error
> detection, which tests how well each method ranks wrong predictions above correct
> ones, plus calibration error. I've already saved all the predictions, so I can
> compute that without retraining.
>
> Then milestone five is the final comparison. Based on what I'm seeing, I don't
> expect a single best method — it'll depend on what's wrong with the image. Thank
> you."

---

## If you run long, cut these first

1. The whole 2:50 challenges section (−30s).
2. The `src/model.py` paragraph at 0:25, keeping only the contrast-augmentation
   point (−15s).

Keep the live demo and `degradation_accuracy.png`. Those two carry the grade: one
is the required demonstration, the other is your main result.

## If you have room for 5 minutes

Add `severity_calibration.png` after the examples figure:

> "Picking the severity levels took real work. My first settings pushed three of the
> five levels down to chance accuracy, which would have given me three useless data
> points. I wrote a script to test a wide range and pick levels that drop gradually.
> All the usable levels sit in a narrow band — round numbers would have missed it."
