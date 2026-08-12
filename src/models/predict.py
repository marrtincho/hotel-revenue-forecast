"""
predict.py
==========
Command-line forecasting: reads a PMS export, produces occupancy forecasts
at each configured horizon, and writes them to CSV.

This is a thin wrapper over `forecast_engine`. All modelling logic lives
there, so the CLI and the Streamlit app cannot drift apart — an earlier
version of this project kept a duplicate copy of the training code here,
and the two silently diverged, which is exactly the bug this structure
prevents.

Usage:
    python src/models/predict.py
    python src/models/predict.py --snapshot-date 2026-01-15
    python src/models/predict.py --files data/raw/pms_exports/export.csv
    python src/models/predict.py --retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast_engine as engine  # noqa: E402


def print_forecast(forecast: pd.DataFrame,
                   snapshot_date: pd.Timestamp,
                   settings: engine.Settings) -> None:
    """Print a per-horizon summary of the 30 days following each lead time."""
    print("\n" + "=" * 72)
    print(f"OCCUPANCY FORECAST — {settings.hotel_name}")
    print(f"Books as at {snapshot_date:%d %B %Y}")
    print("=" * 72)

    for h in settings.horizons:
        window_start = snapshot_date + pd.Timedelta(days=h)
        window_end = window_start + pd.Timedelta(days=30)
        window = forecast[
            (forecast["horizon_days"] == h)
            & (forecast["stay_date"] >= window_start)
            & (forecast["stay_date"] < window_end)
        ]
        if window.empty:
            continue

        print(f"\n{h} days out — arrivals {window_start:%d %b} to {window_end:%d %b %Y}")
        print("-" * 72)
        print(f"{'Room type':<20}{'Cap':>5}{'Books':>8}{'Books %':>10}"
              f"{'Forecast':>10}{'Fcst %':>9}  {'Method'}")
        print("-" * 72)

        summary = window.groupby("room_type_name").agg(
            capacity=("capacity", "first"),
            books=("otb_rooms", "mean"),
            forecast=("predicted_rooms", "mean"),
            method=("method", "first"),
        ).reset_index().sort_values("capacity", ascending=False)

        for _, r in summary.iterrows():
            print(f"{r['room_type_name']:<20}{r['capacity']:>5.0f}"
                  f"{r['books']:>8.1f}{r['books'] / r['capacity'] * 100:>9.1f}%"
                  f"{r['forecast']:>10.1f}{r['forecast'] / r['capacity'] * 100:>8.1f}%"
                  f"  {r['method']}")

        total_cap = settings.total_rooms
        total_books = summary["books"].sum()
        total_fcst = summary["forecast"].sum()
        print("-" * 72)
        print(f"{'HOTEL':<20}{total_cap:>5}{total_books:>8.1f}"
              f"{total_books / total_cap * 100:>9.1f}%"
              f"{total_fcst:>10.1f}{total_fcst / total_cap * 100:>8.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate occupancy forecasts")
    parser.add_argument("--snapshot-date", default=None,
                        help="Date the books were pulled (YYYY-MM-DD). "
                             "Defaults to today.")
    parser.add_argument("--files", nargs="+", default=None,
                        help="PMS export paths. Defaults to everything in "
                             "the configured raw directory.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from scratch instead of loading saved models.")
    args = parser.parse_args()

    settings = engine.Settings()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(settings.raw_pms_dir.glob("*.csv"))
        if not files:
            raise SystemExit(
                f"No PMS exports found in {settings.raw_pms_dir}.\n"
                f"Run: python src/data/generate_synthetic_data.py"
            )

    # Default the snapshot to the export date rather than the wall clock, so
    # the demo works regardless of when someone clones the repository.
    if args.snapshot_date:
        snapshot = pd.Timestamp(args.snapshot_date)
    else:
        snapshot = pd.Timestamp.today().normalize()
        probe = engine.load_pms_files(files, log=lambda _: None)
        live = probe[probe["status"].isin(settings.otb_statuses)]
        if not live.empty and live["stay_date"].max() < snapshot:
            snapshot = live["stay_date"].min().normalize()
            print(f"Note: export contains no dates after today. "
                  f"Using {snapshot:%d %b %Y} as the snapshot date.")

    print("=" * 72)
    print(f"{settings.hotel_name} — {settings.total_rooms} rooms")
    print(f"Snapshot   : {snapshot:%d %B %Y}")
    print(f"Horizons   : {', '.join(f'{h}d' for h in settings.horizons)}")
    print("=" * 72 + "\n")

    forecast = engine.run_forecast(
        pms_file_paths=files,
        snapshot_date=snapshot,
        settings=settings,
        log=print,
        force_retrain=args.retrain,
    )

    if forecast.empty:
        raise SystemExit("No forecast produced — check the export date range.")

    print_forecast(forecast, snapshot, settings)

    print("\n" + "=" * 72)
    print("NOTES")
    print("=" * 72)
    for insight in engine.generate_insights(forecast, snapshot, settings):
        text = insight["text"].replace("**", "")
        print(f"  • {text}")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        settings.output_dir / f"forecast_{snapshot:%Y%m%d}.csv"
    )
    forecast.to_csv(out_path, index=False)
    print(f"\nSaved {len(forecast):,} rows → "
          f"{out_path.relative_to(engine.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
