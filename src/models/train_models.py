"""
train_models.py
===============
Trains the forecasting models and evaluates them against baselines.

The evaluation here is the part that matters. A model that beats nothing is
not a result — so this script always reports the model alongside two
reference points a hotel could implement without any machine learning:

  Baseline 0 — On the books
      Assume no further bookings arrive. This is the floor: it is what the
      property already knows without any forecasting at all.

  Baseline 1 — Median historical pickup
      Add the median pickup observed for that room type and month at that
      lead time. This is essentially what an experienced revenue manager
      does in their head, and it is a genuinely strong baseline.

The split is temporal: the model trains on earlier years and is scored on
the most recent full year it has never seen. A random split would leak
future information into training and produce meaningless scores.

Usage:
    python src/models/train_models.py
    python src/models/train_models.py --val-year 2025
    python src/models/train_models.py --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast_engine as engine  # noqa: E402


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def median_pickup_baseline(train: pd.DataFrame,
                           target: pd.DataFrame,
                           horizon: int) -> np.ndarray:
    """Baseline 1: on the books plus median pickup for that type and month."""
    lookup = train.groupby(["room_type_name", "month"])[f"pickup_{horizon}d"].median()
    global_median = train[f"pickup_{horizon}d"].median()

    pickup = [
        lookup.get((rt, m), global_median)
        for rt, m in zip(target["room_type_name"], target["month"])
    ]
    return np.clip(target[f"otb_rooms_{horizon}d"].to_numpy() + np.array(pickup), 0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models")
    parser.add_argument("--val-year", type=int, default=None,
                        help="Year to hold out for evaluation. Defaults to the "
                             "last complete year in the data.")
    parser.add_argument("--no-save", action="store_true",
                        help="Evaluate only; do not persist models to disk.")
    args = parser.parse_args()

    settings = engine.Settings()

    print("=" * 66)
    print(f"MODEL TRAINING — {settings.hotel_name}")
    print("=" * 66)

    hist = pd.read_csv(settings.features_path, parse_dates=["stay_date"])
    hist = engine.prepare_training_frame(hist, settings)

    # Pick the last complete year as the evaluation set.
    year_counts = hist.groupby(hist["stay_date"].dt.year)["stay_date"].nunique()
    complete_years = year_counts[year_counts >= 360].index.tolist()
    val_year = args.val_year or max(complete_years)

    train = hist[hist["stay_date"].dt.year < val_year]
    val = hist[hist["stay_date"].dt.year == val_year]

    print(f"\nTrain : {train['stay_date'].min():%d %b %Y} → "
          f"{train['stay_date'].max():%d %b %Y}  ({len(train):,} rows)")
    print(f"Test  : {val['stay_date'].min():%d %b %Y} → "
          f"{val['stay_date'].max():%d %b %Y}  ({len(val):,} rows)")

    if train.empty or val.empty:
        raise SystemExit("Not enough history for a temporal split.")

    print("\n" + "-" * 66)
    print("TRAINING")
    print("-" * 66)
    models, _, _, _ = engine.train_models(train, settings, log=lambda m: print(f"  {m}"))

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"EVALUATION — held-out {val_year}")
    print("=" * 66)

    # The single-room type is excluded from headline figures: with a capacity
    # of one, every error is 100% of capacity and it drowns out everything else.
    reportable = [rt for rt in settings.inventory if settings.capacity(rt) > 1]

    rows = []
    for h in settings.horizons:
        for room_type in reportable:
            grp = val[val["room_type_name"] == room_type]
            if grp.empty:
                continue

            y_true = grp["realized_rooms"].to_numpy()
            cap = settings.capacity(room_type)

            b0 = evaluate(y_true, grp[f"otb_rooms_{h}d"].to_numpy())
            b1 = evaluate(y_true, median_pickup_baseline(train, grp, h))

            if (room_type, h) in models:
                model, feats = models[(room_type, h)]
                preds = np.clip(model.predict(grp[feats]), 0, cap * 1.5)
                model_metrics = evaluate(y_true, preds)
                method = "LightGBM"
            else:
                model_metrics = b1
                method = "Median pickup"

            rows.append({
                "horizon": h,
                "room_type": room_type,
                "capacity": cap,
                "method": method,
                "mae_model": model_metrics["mae"],
                "mae_on_books": b0["mae"],
                "mae_median_pickup": b1["mae"],
                "bias_model": model_metrics["bias"],
            })

    results = pd.DataFrame(rows)
    results["improvement_vs_baseline"] = (
        results["mae_median_pickup"] - results["mae_model"]
    )

    for h in settings.horizons:
        sub = results[results["horizon"] == h].sort_values("capacity", ascending=False)
        print(f"\n--- {h} days before arrival ---")
        print(f"{'Room type':<20}{'Cap':>5}{'Model':>9}{'Books':>9}"
              f"{'Median':>9}{'Δ vs base':>11}  {'Method'}")
        print("-" * 78)
        for _, r in sub.iterrows():
            marker = "+" if r["improvement_vs_baseline"] > 0 else " "
            print(f"{r['room_type']:<20}{r['capacity']:>5}"
                  f"{r['mae_model']:>9.1f}{r['mae_on_books']:>9.1f}"
                  f"{r['mae_median_pickup']:>9.1f}"
                  f"{r['improvement_vs_baseline']:>+10.1f}{marker} {r['method']}")

    # Portfolio-level view.
    print("\n" + "=" * 66)
    print("PORTFOLIO SUMMARY (capacity-weighted, single-room type excluded)")
    print("=" * 66)
    print(f"{'Horizon':<10}{'Model MAE':>12}{'Books MAE':>12}"
          f"{'Median MAE':>12}{'Improvement':>14}")
    print("-" * 60)
    for h in settings.horizons:
        sub = results[results["horizon"] == h]
        weights = sub["capacity"] / sub["capacity"].sum()
        m = float((sub["mae_model"] * weights).sum())
        b0 = float((sub["mae_on_books"] * weights).sum())
        b1 = float((sub["mae_median_pickup"] * weights).sum())
        gain = (b1 - m) / b1 * 100 if b1 else 0.0
        print(f"{h:>3}d{'':<7}{m:>12.2f}{b0:>12.2f}{b1:>12.2f}{gain:>13.1f}%")

    results_path = settings.output_dir / "evaluation_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\nDetailed results → {results_path.relative_to(engine.PROJECT_ROOT)}")

    # -------------------------------------------------------------------------
    # Fit final models on all history and persist
    # -------------------------------------------------------------------------
    if not args.no_save:
        print("\n" + "-" * 66)
        print("Refitting on full history for production use")
        print("-" * 66)
        engine.train_and_save(settings, log=lambda m: print(f"  {m}"))

    print("\nDone. Next: python src/models/predict.py")


if __name__ == "__main__":
    main()
