"""
generate_synthetic_data.py
==========================
Generates the synthetic dataset this project runs on.

The model in this repository was developed against a real hotel PMS extract
that cannot be published. To keep the project fully reproducible, this script
generates a synthetic dataset that reproduces the *statistical behaviour* of
hotel booking data — seasonality, day-of-week effects, booking curves,
cancellation and no-show rates, event-driven demand spikes — without
containing any real reservation.

Two artefacts are produced:

  1. data/raw/pms_exports/*.csv
     Row-per-room-per-night exports mimicking a PMS "reservation room stay
     daily" report. These are what the inference pipeline consumes.

  2. data/processed/pickup_features.csv
     The engineered feature table used for training: one row per
     (stay_date, room_type) with realised demand and reconstructed
     on-the-books snapshots at 30/60/90 days before arrival.

Usage:
    python src/data/generate_synthetic_data.py
    python src/data/generate_synthetic_data.py --start 2022-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# -----------------------------------------------------------------------------
# Behavioural parameters
# -----------------------------------------------------------------------------
# These shape the synthetic demand process. They are deliberately expressed as
# interpretable quantities (multipliers, rates, price points) rather than fitted
# coefficients, so a reader can see exactly what the generator assumes.
# -----------------------------------------------------------------------------

# Multiplicative demand factor by calendar month (1.0 = annual average).
# Shoulder and summer months run hot; deep winter is soft.
MONTH_FACTOR = {
    1: 0.67, 2: 1.00, 3: 1.03, 4: 1.02, 5: 1.05, 6: 1.08,
    7: 1.06, 8: 1.12, 9: 1.13, 10: 1.10, 11: 1.02, 12: 0.70,
}

# Multiplicative demand factor by weekday (Mon=0 ... Sun=6).
# Midweek is business-heavy; Sunday is the weakest night.
DOW_FACTOR = {0: 0.93, 1: 1.03, 2: 1.09, 3: 1.06, 4: 1.05, 5: 1.06, 6: 0.79}

# Event impact multipliers, keyed to the ordinal impact level in config.yaml.
EVENT_FACTOR = {"none": 1.00, "low": 1.05, "medium": 1.12, "high": 1.28}

# Per-room-type demand profile:
#   base_occ  — mean occupancy as a fraction of that type's capacity
#   noise_cv  — coefficient of variation of nightly demand
#   adr       — mean achieved ADR in EUR
#   adr_sd    — nightly ADR dispersion
ROOM_PROFILE = {
    "Classic":          dict(base_occ=0.89, noise_cv=0.31, adr=142, adr_sd=26),
    "Comfort":          dict(base_occ=0.85, noise_cv=0.27, adr=131, adr_sd=24),
    "Terrace":          dict(base_occ=0.47, noise_cv=0.65, adr=169, adr_sd=34),
    "Family Plus":      dict(base_occ=0.78, noise_cv=0.55, adr=167, adr_sd=31),
    "Executive":        dict(base_occ=0.45, noise_cv=0.62, adr=252, adr_sd=48),
    "Executive Family": dict(base_occ=0.55, noise_cv=0.53, adr=219, adr_sd=42),
    "Penthouse":        dict(base_occ=0.62, noise_cv=0.70, adr=655, adr_sd=120),
}

# Booking curve: fraction of final realised demand already on the books at a
# given number of days before arrival. Roughly half of all bookings land in the
# final month, which is typical for a city hotel with a strong leisure mix.
BOOKING_CURVE = {90: 0.13, 60: 0.23, 30: 0.40, 14: 0.60, 7: 0.75, 0: 1.00}

# Reservation lifecycle rates.
CANCELLATION_RATE = 0.22   # share of gross bookings later cancelled
NO_SHOW_RATE = 0.018       # share of arrivals that never check in

# Year-level demand multiplier. Introduces the kind of year-over-year regime
# shift that makes naive "same period last year" forecasting fragile — and that
# a pace-aware model has to cope with.
YEAR_FACTOR = {2022: 0.93, 2023: 1.02, 2024: 1.05, 2025: 0.94, 2026: 0.99, 2027: 1.00}

RATE_PLANS = [
    ("BAR", "Best Available Rate"),
    ("NREF", "Non-Refundable"),
    ("CORP", "Corporate Negotiated"),
    ("PKG", "Package Rate"),
    ("GRP", "Group Block"),
]
CHANNELS = [
    ("DIR", "Direct Web"),
    ("OTA", "Online Travel Agency"),
    ("GDS", "Global Distribution System"),
    ("TO", "Tour Operator"),
    ("PHN", "Phone / Walk-in"),
]
SEGMENTS = [
    ("LEIS", "Leisure"),
    ("CORP", "Corporate"),
    ("GRP", "Group"),
    ("MICE", "Meetings & Events"),
]


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_event_lookup(cfg: dict) -> dict:
    """Map (month, day) -> highest event impact active that day."""
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    lookup: dict[tuple[int, int], str] = {}
    for month, d_start, d_end, _name, impact in cfg["events"]:
        for day in range(d_start, d_end + 1):
            key = (month, day)
            if order[impact] > order.get(lookup.get(key, "none"), 0):
                lookup[key] = impact
    return lookup


# -----------------------------------------------------------------------------
# Demand simulation
# -----------------------------------------------------------------------------
def simulate_demand(dates: pd.DatetimeIndex,
                    cfg: dict,
                    rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate realised room-nights per (stay_date, room_type).

    Demand is modelled multiplicatively: a per-type base level scaled by month,
    weekday, year and event factors, then perturbed by lognormal noise. Values
    are capped slightly above physical capacity to mimic the overbooking and
    upgrade behaviour visible in real PMS data.
    """
    inventory = cfg["inventory"]
    events = build_event_lookup(cfg)

    rows = []
    for stay_date in dates:
        m_factor = MONTH_FACTOR[stay_date.month]
        d_factor = DOW_FACTOR[stay_date.dayofweek]
        y_factor = YEAR_FACTOR.get(stay_date.year, 1.0)
        e_factor = EVENT_FACTOR[events.get((stay_date.month, stay_date.day), "none")]

        for room_type, capacity in inventory.items():
            profile = ROOM_PROFILE[room_type]
            expected = (
                capacity
                * profile["base_occ"]
                * m_factor * d_factor * y_factor * e_factor
            )

            # Lognormal multiplicative noise keeps demand strictly positive and
            # right-skewed, matching how occupancy actually disperses.
            sigma = np.sqrt(np.log(1 + profile["noise_cv"] ** 2))
            mu = np.log(max(expected, 0.1)) - 0.5 * sigma ** 2
            realised = rng.lognormal(mu, sigma)

            realised = int(np.clip(round(realised), 0, capacity * 1.25))

            adr = max(
                45.0,
                rng.normal(
                    profile["adr"] * (0.85 + 0.30 * e_factor / EVENT_FACTOR["high"]),
                    profile["adr_sd"],
                ),
            )

            rows.append({
                "stay_date": stay_date,
                "room_type_name": room_type,
                "capacity": capacity,
                "realized_rooms": realised,
                "realized_adr": round(adr, 2),
            })

    df = pd.DataFrame(rows)
    df["realized_revenue"] = (df["realized_rooms"] * df["realized_adr"]).round(2)
    df["realized_occupancy_pct"] = df["realized_rooms"] / df["capacity"]
    return df


def add_otb_snapshots(demand: pd.DataFrame,
                      horizons: list[int],
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Reconstruct on-the-books state at each forecast horizon.

    For every (stay_date, room_type) the realised demand is walked backwards
    along the booking curve, with noise, to produce what the books would have
    looked like N days before arrival. Pickup is the gap the model must learn
    to close.
    """
    df = demand.copy()
    for h in horizons:
        share = BOOKING_CURVE[h]
        # Beta noise around the curve: some dates book early, others late.
        noise = rng.beta(share * 12, (1 - share) * 12, size=len(df))
        blended = 0.55 * share + 0.45 * noise

        otb = np.floor(df["realized_rooms"].to_numpy() * blended)
        otb = np.clip(otb, 0, None).astype(int)

        df[f"otb_rooms_{h}d"] = otb
        df[f"otb_occupancy_pct_{h}d"] = otb / df["capacity"]
        df[f"otb_adr_{h}d"] = (
            df["realized_adr"] * rng.normal(0.97, 0.05, size=len(df))
        ).round(2)
        df[f"otb_revenue_{h}d"] = (otb * df[f"otb_adr_{h}d"]).round(2)
        df[f"pickup_{h}d"] = df["realized_rooms"] - otb

    return df


def add_calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach calendar, holiday and event features."""
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    events_by_day = build_event_lookup(cfg)

    # Named events, for the human-readable columns.
    name_by_day: dict[tuple[int, int], str] = {}
    for month, d_start, d_end, name, _impact in cfg["events"]:
        for day in range(d_start, d_end + 1):
            name_by_day.setdefault((month, day), name)

    holidays_nat = {tuple(x) for x in cfg["holidays_national"]}
    holidays_reg = {tuple(x) for x in cfg["holidays_regional"]}
    peak_months = set(cfg["peak_months"])

    d = df["stay_date"]
    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_month"] = d.dt.day
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_year"] = d.dt.dayofyear
    df["year"] = d.dt.year
    df["days_to_year_end"] = (
        pd.to_datetime(d.dt.year.astype(str) + "-12-31") - d
    ).dt.days

    md = list(zip(d.dt.month, d.dt.day))
    df["is_holiday_national"] = [int(x in holidays_nat) for x in md]
    df["is_holiday_regional"] = [int(x in holidays_reg) for x in md]
    df["is_peak_season"] = d.dt.month.isin(peak_months).astype(int)

    impacts = [events_by_day.get(x, "none") for x in md]
    df["event_today"] = [int(i != "none") for i in impacts]
    df["event_names"] = [name_by_day.get(x, "") for x in md]
    df["event_max_impact"] = impacts
    df["event_max_impact_enc"] = [order[i] for i in impacts]

    # Days until the next event, looking up to 90 days ahead.
    days_to_next, next_impact = [], []
    for stay_date in d:
        gap, impact = 999, "none"
        for lookahead in range(1, 91):
            future = stay_date + pd.Timedelta(days=lookahead)
            key = (future.month, future.day)
            if key in events_by_day:
                gap, impact = lookahead, events_by_day[key]
                break
        days_to_next.append(gap)
        next_impact.append(impact)

    df["days_to_next_event"] = days_to_next
    df["next_event_impact"] = next_impact
    df["next_event_impact_enc"] = [order[i] for i in next_impact]
    return df


# -----------------------------------------------------------------------------
# PMS export generation
# -----------------------------------------------------------------------------
def generate_pms_rows(demand: pd.DataFrame,
                      cfg: dict,
                      snapshot_date: pd.Timestamp,
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Expand aggregated demand into individual reservation-night rows, in the
    shape a PMS "reservation room stay daily" report would deliver.

    Reservations are given plausible lifecycles: most are honoured, a fifth are
    cancelled, a small tail no-shows. Status is assigned relative to the
    snapshot date so a fresh export contains live future bookings.
    """
    hotel = cfg["hotel"]
    rows = []
    reservation_seq = 100000

    # Interpolator over the booking curve: how much of final demand is
    # typically on the books this many days before arrival.
    curve_days = sorted(BOOKING_CURVE, reverse=True)
    curve_vals = [BOOKING_CURVE[d] for d in curve_days]

    def booked_share(days_out: float) -> float:
        """Fraction of final demand on the books `days_out` days before arrival."""
        if days_out <= 0:
            return 1.0
        if days_out >= curve_days[0]:
            # Beyond the far end of the curve, decay towards zero.
            return curve_vals[0] * (curve_days[0] / days_out)
        return float(np.interp(days_out, curve_days[::-1], curve_vals[::-1]))

    for _, row in demand.iterrows():
        stay_date = row["stay_date"]
        room_type = row["room_type_name"]
        n_rooms = int(row["realized_rooms"])
        if n_rooms <= 0:
            continue

        # Inflate to gross bookings so that cancellations land us back on the
        # realised figure.
        gross = int(np.ceil(n_rooms / (1 - CANCELLATION_RATE)))

        # For future arrivals, only the share of demand that would realistically
        # have booked by the snapshot date exists yet. Generating all of it
        # would hand the model a book position no real export ever shows.
        days_out = (stay_date - snapshot_date).days
        if days_out > 0:
            gross = int(np.ceil(gross * booked_share(days_out)))
            if gross <= 0:
                continue

        for _ in range(gross):
            reservation_seq += 1
            # Booking date must fall on or before the snapshot for future stays.
            max_lead = days_out if days_out > 0 else 330
            lead_days = int(np.clip(rng.exponential(38), max(days_out, 0), 330))
            if days_out > 0:
                lead_days = max(lead_days, days_out)
            booking_dt = stay_date - pd.Timedelta(days=lead_days)
            nights = int(np.clip(rng.geometric(0.42), 1, 14))
            checkin = stay_date - pd.Timedelta(days=int(rng.integers(0, nights)))
            checkout = checkin + pd.Timedelta(days=nights)

            draw = rng.random()
            if draw < CANCELLATION_RATE:
                status = "Cancelled"
            elif draw < CANCELLATION_RATE + NO_SHOW_RATE:
                status = "NoShow"
            elif stay_date > snapshot_date:
                status = "Reserved"
            elif checkin <= snapshot_date <= checkout:
                status = "InHouse"
            else:
                status = "CheckedOut"

            adr = max(40.0, rng.normal(row["realized_adr"], row["realized_adr"] * 0.16))
            rate_code, rate_name = RATE_PLANS[rng.integers(len(RATE_PLANS))]
            chan_code, chan_name = CHANNELS[rng.integers(len(CHANNELS))]
            seg_code, seg_name = SEGMENTS[rng.integers(len(SEGMENTS))]

            room_rev = round(adr * nights, 2)
            extras = round(rng.uniform(0, 45) * nights, 2)
            other = round(rng.uniform(0, 18) * nights, 2)

            rows.append([
                hotel["code"], hotel["name"],
                f"{hotel['code']}{reservation_seq:08d}", 1,
                checkin.strftime("%d/%m/%Y"), checkout.strftime("%d/%m/%Y"),
                room_rev, extras, other, round(room_rev + extras + other, 2),
                booking_dt.strftime("%d/%m/%Y %H:%M:%S"),
                f"L{rng.integers(1, 4)}", status,
                checkin.strftime("%d/%m/%Y"), checkout.strftime("%d/%m/%Y"), nights,
                room_rev, extras, other, round(room_rev + extras + other, 2),
                stay_date.strftime("%d/%m/%Y"),
                room_type[:3].upper(), room_type,
                int(rng.integers(1, 3)), 0,
                int(rng.integers(0, 2)), 0,
                round(adr, 2),
                rate_code, rate_name, "", "",
                "", "", seg_code, seg_name,
                chan_code, chan_name, chan_code, chan_name,
            ])

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic hotel dataset")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--future-end", default="2026-12-31",
                        help="End of the forward-looking PMS export window.")
    parser.add_argument("--snapshot-date", default="2026-01-15",
                        help="'Today' for the generated PMS export.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_config()

    print("=" * 62)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 62)
    print(f"Property : {cfg['hotel']['name']} ({sum(cfg['inventory'].values())} rooms)")
    print(f"History  : {args.start} → {args.end}")
    print(f"Snapshot : {args.snapshot_date}")
    print()

    # --- Historical feature table -------------------------------------------
    hist_dates = pd.date_range(args.start, args.end, freq="D")
    print(f"[1/4] Simulating demand for {len(hist_dates)} days...")
    demand = simulate_demand(hist_dates, cfg, rng)

    print("[2/4] Reconstructing on-the-books snapshots...")
    features = add_otb_snapshots(demand, cfg["horizons"], rng)
    features = add_calendar_features(features, cfg)

    features_path = PROJECT_ROOT / cfg["paths"]["features"]
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    print(f"      → {features_path.relative_to(PROJECT_ROOT)} "
          f"({len(features):,} rows × {features.shape[1]} cols)")

    # --- Forward-looking PMS export -----------------------------------------
    snapshot = pd.Timestamp(args.snapshot_date)
    print("[3/4] Simulating forward demand for PMS export...")
    future_dates = pd.date_range(
        pd.Timestamp(args.end) + pd.Timedelta(days=1), args.future_end, freq="D"
    )
    future_demand = simulate_demand(future_dates, cfg, rng)

    print("[4/4] Expanding to reservation-night rows...")
    export_window = pd.concat([
        demand[demand["stay_date"] >= snapshot - pd.Timedelta(days=120)],
        future_demand,
    ])
    pms = generate_pms_rows(export_window, cfg, snapshot, rng)

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_pms"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    export_path = raw_dir / (
        f"{cfg['hotel']['code']}_reservationRoomStayDaily_"
        f"{snapshot.strftime('%Y%m%d')}_090000.csv"
    )
    # Headerless, matching the PMS report format the pipeline expects.
    pms.to_csv(export_path, index=False, header=False)
    print(f"      → {export_path.relative_to(PROJECT_ROOT)} ({len(pms):,} rows)")

    # --- Summary ------------------------------------------------------------
    total_cap = sum(cfg["inventory"].values())
    hotel_occ = (
        features.groupby("stay_date")["realized_rooms"].sum() / total_cap
    ).mean()
    print()
    print("=" * 62)
    print(f"Mean hotel occupancy : {hotel_occ * 100:.1f}%")
    print(f"Mean ADR             : {features['realized_adr'].mean():.2f} "
          f"{cfg['hotel']['currency']}")
    print(f"Live bookings in export : "
          f"{(pms.iloc[:, 12] == 'Reserved').sum():,}")
    print("=" * 62)
    print("\nDone. Next: python src/models/train_models.py")


if __name__ == "__main__":
    main()
