"""
forecast_engine.py
==================
Core forecasting logic, shared by the CLI (`predict.py`) and the Streamlit
app (`src/app/app.py`).

Design notes
------------
Everything property-specific — hotel name, room inventory, event calendar,
modelling strategy, hyperparameters — is read from `config/config.yaml`.
The module contains no hardcoded hotel data, so pointing it at a different
property is a config change, not a code change.

Two decisions in here are worth calling out, because both were made in
response to observed failure modes rather than chosen up front:

1. **The target is absolute room-nights, not occupancy percentage.**
   Occupancy is a ratio against a fixed denominator, so the two are
   informationally identical — but predicting counts keeps the error metric
   directly interpretable ("we were 8 rooms out") and avoids awkward
   behaviour on small room types where a single room is a large percentage.

2. **Monotonic constraints are enforced on the on-the-books features.**
   An unconstrained model trained to convergence learned to lean on
   calendar features and became nearly insensitive to current bookings —
   producing near-identical forecasts regardless of actual demand. Forcing
   "more rooms on the books never lowers the forecast" restores the
   business logic the model is supposed to encode. See docs/METHODOLOGY.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

IMPACT_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

# PMS report schema. The export arrives without a header row, so column names
# are supplied positionally.
PMS_COLS = [
    "hotel_code", "hotel_name", "reservation_id", "num_rooms",
    "checkin_date_original", "checkout_date_original",
    "total_room_revenue", "total_extras_revenue",
    "total_other_revenue", "total_revenue",
    "booking_datetime", "reservation_line_id", "status",
    "checkin_date_current", "checkout_date_current", "num_nights",
    "room_revenue_stay_total", "extras_revenue_stay_total",
    "other_revenue_stay_total", "total_revenue_stay_total",
    "stay_date", "room_type_code", "room_type_name", "num_guests",
    "num_juniors", "num_children", "num_babies_cots",
    "room_adr_actual_night", "rate_code", "rate_name",
    "promo_code", "promo_name", "company_id", "company_name",
    "segment_code", "segment_name", "source_code", "source_name",
    "channel_code", "channel_name",
]

# Calendar and event features shared across all horizons.
COMMON_FEATS = [
    "day_of_week", "is_weekend", "month", "quarter",
    "week_of_year", "day_of_year", "year",
    "is_holiday_national", "is_holiday_regional",
    "is_peak_season",
    "event_today", "event_max_impact_enc",
    "days_to_next_event", "next_event_impact_enc",
    "lag_364d_rooms", "lag_365d_rooms",
]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Settings:
    """Flattened, validated view over config.yaml."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.raw = cfg

        self.hotel_name = cfg["hotel"]["name"]
        self.hotel_code = cfg["hotel"]["code"]
        self.currency = cfg["hotel"]["currency"]

        self.inventory: dict[str, int] = dict(cfg["inventory"])
        self.total_rooms = sum(self.inventory.values())

        self.use_model = set(cfg["strategy"]["use_model"])
        self.use_baseline = set(cfg["strategy"]["use_baseline"])
        self.use_otb_only = set(cfg["strategy"]["use_otb_only"])

        self.horizons: list[int] = list(cfg["horizons"])
        self.otb_statuses: list[str] = list(cfg["otb_statuses"])

        m = cfg["model"]
        self.early_stopping_rounds = m["early_stopping_rounds"]
        self.internal_val_fraction = m["internal_val_fraction"]
        self.lgbm_params = {
            k: v for k, v in m.items()
            if k not in {"early_stopping_rounds", "internal_val_fraction"}
        }
        self.lgbm_params["verbose"] = -1

        self.events = cfg["events"]
        self.holidays_national = {tuple(x) for x in cfg["holidays_national"]}
        self.holidays_regional = {tuple(x) for x in cfg["holidays_regional"]}
        self.peak_months = set(cfg["peak_months"])

        p = cfg["paths"]
        self.features_path = PROJECT_ROOT / p["features"]
        self.raw_pms_dir = PROJECT_ROOT / p["raw_pms"]
        self.output_dir = PROJECT_ROOT / p["output"]
        self.models_dir = PROJECT_ROOT / p["models"]

    def capacity(self, room_type: str) -> int:
        return self.inventory[room_type]

    def method_for(self, room_type: str) -> str:
        if room_type in self.use_model:
            return "LightGBM"
        if room_type in self.use_baseline:
            return "Median pickup"
        return "On the books"


# -----------------------------------------------------------------------------
# PMS ingestion
# -----------------------------------------------------------------------------
def load_pms_files(file_paths: list, log=print) -> pd.DataFrame:
    """Read one or more headerless PMS exports into a single frame."""
    frames = []
    for path in file_paths:
        path = Path(path)
        try:
            frames.append(pd.read_csv(
                path, header=None, names=PMS_COLS,
                parse_dates=["stay_date", "booking_datetime"],
                dayfirst=True,
            ))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            log(f"Could not read {path.name}: {exc}")

    if not frames:
        raise FileNotFoundError("No PMS export could be read.")

    combined = pd.concat(frames, ignore_index=True)
    log(f"Loaded {len(file_paths)} export(s), {len(combined):,} rows")
    return combined


def build_otb(pms: pd.DataFrame,
              snapshot_date: pd.Timestamp,
              settings: Settings,
              log=print) -> pd.DataFrame:
    """
    Aggregate live reservations into on-the-books room-nights per
    (stay_date, room_type), filling absent combinations with zero.
    """
    active = pms[
        pms["status"].isin(settings.otb_statuses)
        & (pms["stay_date"] >= snapshot_date)
    ].copy()

    if active.empty:
        raise ValueError(
            f"No live reservations ({'/'.join(settings.otb_statuses)}) found on "
            f"or after {snapshot_date.date()}. The export may be historical "
            f"only, or the snapshot date may be too late."
        )

    otb = active.groupby(["stay_date", "room_type_name"]).agg(
        otb_rooms=("num_rooms", "sum"),
        otb_revenue=(
            "room_adr_actual_night",
            lambda s: float((s * active.loc[s.index, "num_rooms"]).sum()),
        ),
    ).reset_index()

    full_index = pd.MultiIndex.from_product(
        [pd.date_range(snapshot_date, otb["stay_date"].max(), freq="D"),
         list(settings.inventory)],
        names=["stay_date", "room_type_name"],
    )
    otb = (
        otb.set_index(["stay_date", "room_type_name"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    log(f"On the books: {len(active):,} reservation-nights across "
        f"{otb['stay_date'].nunique()} dates")
    return otb


# -----------------------------------------------------------------------------
# Calendar features
# -----------------------------------------------------------------------------
def build_calendar_features(dates: pd.DatetimeIndex,
                            settings: Settings) -> pd.DataFrame:
    """Build calendar, holiday and event features for a date range."""
    impact_by_day: dict[tuple[int, int], str] = {}
    name_by_day: dict[tuple[int, int], str] = {}
    for month, d_start, d_end, name, impact in settings.events:
        for day in range(d_start, d_end + 1):
            key = (month, day)
            if IMPACT_ORDER[impact] > IMPACT_ORDER.get(impact_by_day.get(key, "none"), 0):
                impact_by_day[key] = impact
            name_by_day.setdefault(key, name)

    df = pd.DataFrame({"stay_date": dates})
    d = df["stay_date"]

    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_month"] = d.dt.day
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_year"] = d.dt.dayofyear
    df["year"] = d.dt.year

    md = list(zip(d.dt.month, d.dt.day))
    df["is_holiday_national"] = [int(x in settings.holidays_national) for x in md]
    df["is_holiday_regional"] = [int(x in settings.holidays_regional) for x in md]
    df["is_peak_season"] = d.dt.month.isin(settings.peak_months).astype(int)

    impacts = [impact_by_day.get(x, "none") for x in md]
    df["event_today"] = [int(i != "none") for i in impacts]
    df["event_names"] = [name_by_day.get(x, "") for x in md]
    df["event_max_impact"] = impacts
    df["event_max_impact_enc"] = [IMPACT_ORDER[i] for i in impacts]

    days_to_next, next_impact, next_name = [], [], []
    for stay_date in d:
        gap, impact, name = 999, "none", ""
        for lookahead in range(1, 91):
            future = stay_date + pd.Timedelta(days=lookahead)
            key = (future.month, future.day)
            if key in impact_by_day:
                gap, impact = lookahead, impact_by_day[key]
                name = name_by_day.get(key, "")
                break
        days_to_next.append(gap)
        next_impact.append(impact)
        next_name.append(name)

    df["days_to_next_event"] = days_to_next
    df["next_event_impact"] = next_impact
    df["next_event_name"] = next_name
    df["next_event_impact_enc"] = [IMPACT_ORDER[i] for i in next_impact]
    return df


# -----------------------------------------------------------------------------
# Lag features
# -----------------------------------------------------------------------------
def _add_lags(df: pd.DataFrame,
              reference: pd.DataFrame,
              settings: Settings) -> pd.DataFrame:
    """
    Attach same-period-last-year features.

    364 days back lands on the same weekday, which matters because weekday
    is one of the strongest demand signals; 365 lands on the same calendar
    date, which matters for fixed-date events. Both are provided and the
    model decides which is useful where. LightGBM handles the missing values
    for the first year of history natively.
    """
    rooms = reference.set_index(["room_type_name", "stay_date"])["realized_rooms"]

    for lag_days, col in [(364, "lag_364d_rooms"), (365, "lag_365d_rooms")]:
        df[col] = [
            rooms.get((rt, sd - pd.Timedelta(days=lag_days)), np.nan)
            for rt, sd in zip(df["room_type_name"], df["stay_date"])
        ]

    for h in settings.horizons:
        col = f"otb_rooms_{h}d"
        if col in reference.columns:
            prior_otb = reference.set_index(["room_type_name", "stay_date"])[col]
            df[f"lag_364d_otb_{h}d"] = [
                prior_otb.get((rt, sd - pd.Timedelta(days=364)), np.nan)
                for rt, sd in zip(df["room_type_name"], df["stay_date"])
            ]
            # Pace: this year's book position against the same point last year.
            # The +0.5 keeps the ratio finite when last year's books were empty.
            df[f"pace_ratio_{h}d"] = df[col] / (df[f"lag_364d_otb_{h}d"] + 0.5)
            df[f"otb_vs_realized_lag_{h}d"] = df[col] / (df["lag_364d_rooms"] + 0.5)
        else:
            df[f"lag_364d_otb_{h}d"] = np.nan
            df[f"pace_ratio_{h}d"] = np.nan
            df[f"otb_vs_realized_lag_{h}d"] = np.nan

    return df


def horizon_features(h: int, columns) -> tuple[list[str], list[int]]:
    """
    Feature list and matching monotonic constraints for one horizon.

    Constraint of +1 means "increasing this feature must never decrease the
    prediction". Applied to the three demand-signal features so the model
    cannot learn a non-monotonic relationship with current bookings.

    Derived quantities like OTB revenue and OTB occupancy percentage are
    deliberately excluded: they are near-collinear with the room count and
    only dilute its importance.
    """
    candidates = [
        (f"otb_rooms_{h}d", 1),
        (f"lag_364d_otb_{h}d", 0),
        (f"pace_ratio_{h}d", 1),
        (f"otb_vs_realized_lag_{h}d", 1),
    ]
    feats = COMMON_FEATS + [f for f, _ in candidates if f in columns]
    monotone = [0] * len(COMMON_FEATS) + [c for f, c in candidates if f in columns]
    return feats, monotone


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def prepare_training_frame(hist: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Enrich the historical feature table with lag and pace features."""
    hist = hist.copy()
    hist["capacity"] = hist["room_type_name"].map(settings.inventory)
    hist["month"] = hist["stay_date"].dt.month

    if "event_max_impact_enc" not in hist.columns:
        hist["event_max_impact_enc"] = (
            hist["event_max_impact"].fillna("none").map(IMPACT_ORDER).fillna(0).astype(int)
        )
    if "next_event_impact_enc" not in hist.columns:
        hist["next_event_impact_enc"] = (
            hist["next_event_impact"].fillna("none").map(IMPACT_ORDER).fillna(0).astype(int)
        )

    hist = hist.sort_values(["room_type_name", "stay_date"]).reset_index(drop=True)
    return _add_lags(hist, hist, settings)


def train_models(hist: pd.DataFrame,
                 settings: Settings,
                 log=print) -> tuple[dict, dict, pd.DataFrame, dict]:
    """
    Train one model per (room type, horizon) for the types configured to use
    the model, and build the median-pickup lookup used by everything else.

    Validation is a temporal holdout — the most recent slice of each room
    type's history — never a random split. A random split would let the model
    see future dates while predicting past ones, which inflates scores and
    tells you nothing about forward performance.
    """
    hist = prepare_training_frame(hist, settings)

    baseline_lookup = {
        h: hist.groupby(["room_type_name", "month"])[f"pickup_{h}d"].median()
        for h in settings.horizons
    }

    models: dict[tuple[str, int], tuple] = {}
    metrics: dict[tuple[str, int], dict] = {}

    for room_type in settings.use_model:
        subset = (
            hist[hist["room_type_name"] == room_type]
            .sort_values("stay_date")
            .reset_index(drop=True)
        )
        n_val = max(30, int(len(subset) * settings.internal_val_fraction))
        train_df, val_df = subset.iloc[:-n_val], subset.iloc[-n_val:]

        for h in settings.horizons:
            feats, monotone = horizon_features(h, subset.columns)

            model = lgb.LGBMRegressor(
                monotone_constraints=monotone,
                monotone_constraints_method="advanced",
                **settings.lgbm_params,
            )
            model.fit(
                train_df[feats], train_df["realized_rooms"],
                eval_set=[(val_df[feats], val_df["realized_rooms"])],
                callbacks=[
                    lgb.early_stopping(settings.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            cap = settings.capacity(room_type)
            preds = np.clip(model.predict(val_df[feats]), 0, cap * 1.5)
            mae = float(np.mean(np.abs(preds - val_df["realized_rooms"])))

            models[(room_type, h)] = (model, feats)
            metrics[(room_type, h)] = {
                "mae_rooms": round(mae, 2),
                "mae_pct_of_capacity": round(mae / cap * 100, 2),
                "trees": model.best_iteration_,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
            }
            log(f"{room_type} @ {h}d — {model.best_iteration_} trees, "
                f"holdout MAE {mae:.1f} rooms")

    return models, baseline_lookup, hist, metrics


def train_and_save(settings: Settings, log=print) -> dict:
    """Train and persist to disk. Returns the metrics dictionary."""
    log("Loading historical features...")
    hist = pd.read_csv(settings.features_path, parse_dates=["stay_date"])

    log("Training...")
    models, baseline_lookup, hist_enriched, metrics = train_models(
        hist, settings, log=log
    )

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "baseline_lookup": baseline_lookup,
            "hist_enriched": hist_enriched,
            "metrics": metrics,
            "trained_at": pd.Timestamp.now(),
            "inventory": settings.inventory,
        },
        settings.models_dir / "trained_models.joblib",
    )
    log("Saved to data/models/trained_models.joblib")
    return metrics


def load_trained(settings: Settings, log=print):
    """Load persisted models, or None if training hasn't run yet."""
    path = settings.models_dir / "trained_models.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    log(f"Loaded models trained {pd.Timestamp(bundle['trained_at']):%d %b %Y %H:%M}")
    return bundle


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def build_inference_frame(otb: pd.DataFrame,
                          calendar: pd.DataFrame,
                          hist_enriched: pd.DataFrame,
                          settings: Settings) -> pd.DataFrame:
    """Join on-the-books, calendar and lag features into a predict-ready frame."""
    df = otb.merge(calendar, on="stay_date", how="left")
    df["capacity"] = df["room_type_name"].map(settings.inventory)

    # A live export gives one book position per date. The horizon-specific
    # columns therefore all carry the same value; what differs per horizon is
    # the model applied to them, each trained on how books at that lead time
    # historically converted into final demand.
    for h in settings.horizons:
        df[f"otb_rooms_{h}d"] = df["otb_rooms"]

    df = _add_lags(df, hist_enriched, settings)
    df["month"] = df["stay_date"].dt.month
    return df


def predict(df: pd.DataFrame,
            models: dict,
            baseline_lookup: dict,
            snapshot_date: pd.Timestamp,
            settings: Settings) -> pd.DataFrame:
    """Apply the configured method per room type and assemble the forecast."""
    records = []

    for room_type in settings.inventory:
        cap = settings.capacity(room_type)
        subset = df[df["room_type_name"] == room_type]
        if subset.empty:
            continue

        for h in settings.horizons:
            window = subset[
                subset["stay_date"] >= snapshot_date + pd.Timedelta(days=h)
            ].copy()
            if window.empty:
                continue

            if room_type in settings.use_model and (room_type, h) in models:
                model, feats = models[(room_type, h)]
                preds = model.predict(window[feats])

            elif room_type in settings.use_baseline:
                lookup = baseline_lookup[h]
                fallback = (
                    lookup.xs(room_type, level=0).median()
                    if room_type in lookup.index.get_level_values(0) else 0.0
                )
                pickup = [
                    lookup.get((room_type, m), fallback) for m in window["month"]
                ]
                preds = window[f"otb_rooms_{h}d"].to_numpy() + np.array(pickup)

            else:
                preds = window[f"otb_rooms_{h}d"].to_numpy()

            preds = np.clip(preds, 0, cap * 1.5)

            records.append(pd.DataFrame({
                "stay_date": window["stay_date"].values,
                "room_type_name": room_type,
                "capacity": cap,
                "horizon_days": h,
                "snapshot_date": snapshot_date,
                "otb_rooms": window["otb_rooms"].values,
                "otb_occupancy_pct": (window["otb_rooms"].values / cap * 100).round(1),
                "predicted_rooms": np.round(preds, 1),
                "predicted_occupancy_pct": np.round(preds / cap * 100, 1),
                "method": settings.method_for(room_type),
                "pace_ratio": window.get(f"pace_ratio_{h}d", pd.Series(np.nan)).values,
            }))

    if not records:
        return pd.DataFrame()

    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["horizon_days", "stay_date", "room_type_name"])
        .reset_index(drop=True)
    )


def run_forecast(pms_file_paths: list,
                 snapshot_date: pd.Timestamp,
                 settings: Settings | None = None,
                 log=print,
                 force_retrain: bool = False) -> pd.DataFrame:
    """End-to-end: read exports, build features, predict."""
    settings = settings or Settings()

    pms = load_pms_files(pms_file_paths, log=log)
    otb = build_otb(pms, snapshot_date, settings, log=log)

    bundle = None if force_retrain else load_trained(settings, log=log)
    if bundle is None:
        log("No saved models found — training now...")
        train_and_save(settings, log=log)
        bundle = load_trained(settings, log=log)

    calendar = build_calendar_features(
        pd.date_range(snapshot_date, otb["stay_date"].max(), freq="D"), settings
    )
    frame = build_inference_frame(otb, calendar, bundle["hist_enriched"], settings)

    log("Predicting...")
    return predict(
        frame, bundle["models"], bundle["baseline_lookup"], snapshot_date, settings
    )


# -----------------------------------------------------------------------------
# Commentary for the UI
# -----------------------------------------------------------------------------
def generate_insights(forecast: pd.DataFrame,
                      snapshot_date: pd.Timestamp,
                      settings: Settings) -> list[dict]:
    """
    Turn the forecast into a short list of plain-language observations.

    Deliberately conservative: these flag where a revenue manager might want
    to look, they do not recommend a price. Pricing depends on competitor
    rates, group blocks and commercial strategy that this model never sees.
    """
    insights: list[dict] = []
    headline_types = sorted(
        settings.use_model | settings.use_baseline,
        key=lambda rt: -settings.capacity(rt),
    )[:3]

    for h in settings.horizons:
        window_start = snapshot_date + pd.Timedelta(days=h)
        window = forecast[
            (forecast["horizon_days"] == h)
            & (forecast["stay_date"] >= window_start)
            & (forecast["stay_date"] < window_start + pd.Timedelta(days=30))
        ]
        if window.empty:
            continue

        for room_type in headline_types:
            grp = window[window["room_type_name"] == room_type]
            if grp.empty:
                continue

            predicted = grp["predicted_occupancy_pct"].mean()
            on_books = grp["otb_occupancy_pct"].mean()

            if predicted >= 90:
                insights.append({
                    "level": "success",
                    "text": (
                        f"**{room_type}** at {h} days out is tracking to "
                        f"**{predicted:.0f}%**. Worth reviewing rate on the "
                        f"dates still open."
                    ),
                })
            elif predicted <= 45:
                insights.append({
                    "level": "warning",
                    "text": (
                        f"**{room_type}** at {h} days out is tracking to only "
                        f"**{predicted:.0f}%** (currently {on_books:.0f}% on the "
                        f"books). Check for a known cause before acting."
                    ),
                })

    # Flag types where books already exceed physical capacity — usually an
    # add-on product or overbooking policy rather than a data error, but the
    # user should know the number is not a bug.
    over = forecast[forecast["otb_occupancy_pct"] > 100]["room_type_name"].unique()
    for room_type in over:
        insights.append({
            "level": "info",
            "text": (
                f"**{room_type}** shows more than 100% on the books. This is "
                f"expected where a type carries an add-on product or an "
                f"overbooking allowance."
            ),
        })

    if not insights:
        insights.append({
            "level": "info",
            "text": "Nothing unusual in this forecast — demand is tracking close to normal.",
        })

    return insights
