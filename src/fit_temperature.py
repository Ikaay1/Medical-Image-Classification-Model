"""Fit the temperature-scaling parameter T (Milestone 3).

Usage:
    python -m src.fit_temperature

T is fitted by minimising negative log-likelihood on one half of the
validation split and reported on the other half, so the benefit is never
measured on the data that produced it. The test set is never touched.

Outputs:
    outputs/metrics/temperature.json
"""

from __future__ import annotations

import json

import numpy as np

from . import config, uncertainty as U


def main() -> None:
    path = config.PREDICTION_DIR / "logits_val.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python -m src.evaluate")

    with np.load(path) as d:
        logits, labels = d["logits"], d["labels"]

    fit_idx, eval_idx = U.split_validation(len(labels))
    print(f"Validation split: {len(fit_idx):,} to fit T, {len(eval_idx):,} held out\n")

    temperature = U.fit_temperature(logits[fit_idx], labels[fit_idx])

    record = {"temperature": temperature, "n_fit": int(len(fit_idx)), "n_eval": int(len(eval_idx))}
    print(f"Fitted T = {temperature:.4f}")
    print("  T > 1 means the model was overconfident and is being softened.\n")

    for split_name, idx in (("fit half", fit_idx), ("held-out half", eval_idx)):
        lg, lb = logits[idx], labels[idx]
        before = U.softmax_probs(lg)
        after = U.temperature_probs(lg, temperature)

        acc = float((before.argmax(1) == lb).mean())
        acc_after = float((after.argmax(1) == lb).mean())
        stats = {
            "accuracy": acc,
            "accuracy_after": acc_after,
            "nll_before": U.negative_log_likelihood(lg, lb, 1.0),
            "nll_after": U.negative_log_likelihood(lg, lb, temperature),
            "mean_confidence_before": float(before.max(1).mean()),
            "mean_confidence_after": float(after.max(1).mean()),
        }
        record[split_name.replace(" ", "_").replace("-", "_")] = stats

        print(f"{split_name}:")
        print(f"  accuracy          {acc:.4f}  (after scaling {acc_after:.4f})")
        print(f"  mean confidence   {stats['mean_confidence_before']:.4f} -> {stats['mean_confidence_after']:.4f}")
        print(f"  NLL               {stats['nll_before']:.4f} -> {stats['nll_after']:.4f}")
        print(f"  confidence - accuracy  {stats['mean_confidence_before']-acc:+.4f} -> {stats['mean_confidence_after']-acc:+.4f}\n")

    config.TEMPERATURE_PATH.write_text(json.dumps(record, indent=2))
    print(f"Temperature -> {config.TEMPERATURE_PATH}")


if __name__ == "__main__":
    main()
