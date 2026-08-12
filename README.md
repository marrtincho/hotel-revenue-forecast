# Hotel Revenue Forecasting

Forecasting final occupancy by room type at 30, 60 and 90 days before arrival,
using pickup methodology and gradient boosting.

Hotels know how many rooms are booked today. What they need to know is how many
will be booked by arrival — the gap between those two numbers is *pickup*, and
predicting it well is what lets a revenue manager price a date three months out
with any confidence.

This project builds that forecast end to end: raw property management system
exports in, per-room-type occupancy forecasts out, with a Streamlit interface
for the people who actually make pricing decisions.

---

## Why this is not a straightforward regression problem

**The target moves under you.** A date's final occupancy depends on bookings
that have not happened yet. Training on historical outcomes means reconstructing
what the books looked like at each lead time — a snapshot problem, not a
row-per-observation problem.

**Random splits leak the future.** Booking behaviour is strongly autocorrelated
across dates. A random train/test split lets the model see September while
predicting August, which inflates every metric and tells you nothing about
forward performance. All validation here is temporal.

**Small room types are mostly noise.** A 98-room type has enough nightly
variation to learn from. A 2-room type does not. Applying one model uniformly
across all types produces confident-looking nonsense on the small ones.

**The obvious baseline is strong.** "Add the median pickup for this room type
and month" is roughly what an experienced revenue manager does mentally, and it
is hard to beat. Any model that does not clear it is not worth deploying.

---

## Results

Held-out year, never seen during training. Error is mean absolute error in
rooms per night — the unit a revenue manager can act on.

| Lead time | Model | No further bookings | Median pickup | Improvement |
|-----------|-------|---------------------|---------------|-------------|
| 30 days   | 6.1   | 30.3                | 7.8           | **22.0%**   |
| 60 days   | 8.2   | 38.9                | 9.9           | **16.9%**   |
| 90 days   | 8.9   | 43.9                | 10.8          | **17.2%**   |

*Capacity-weighted across room types. Single-room type excluded — with a
capacity of one, every error is 100% of capacity and it distorts any average.*

The honest version of that table: the model wins clearly on high-volume room
types and is **level with the baseline on low-volume ones**. Those types are
configured to use the baseline in production rather than pretending the model
adds something. Per-type results are in `data/output/evaluation_results.csv`
and in the app's performance page.

---

## Two bugs worth documenting

Both were found by interrogating model behaviour rather than by looking at
aggregate metrics, which is the point of including them here.

### The model stopped listening to demand

Forecasts came back at 90–100% occupancy almost regardless of the date or the
current book position. Aggregate error looked acceptable, so nothing flagged it.

The diagnostic was a controlled sensitivity test: hold every feature fixed,
sweep on-the-books from empty to full, and watch the prediction. A working
model should climb monotonically. This one lurched — 83%, then 112%, then 94%,
then 96%.

The cause was training to a fixed 1000 trees with no early stopping. The model
had over-fit into leaning on calendar features and had effectively stopped using
current bookings. Three changes fixed it:

- **Temporal early stopping** against a held-out recent slice per room type
- **Monotonic constraints** on the demand features, so more rooms on the books
  can never lower the forecast — encoding a business rule the model was free to
  violate
- **Dropping collinear derived features** (OTB revenue, OTB occupancy
  percentage) that diluted the room-count signal without adding information

After the fix the sweep is strictly monotonic: 57% → 74% → 78% → ... → 88%.

### Two copies of the training code

The command-line tool and the app each carried their own copy of the training
logic. One had early stopping; the other did not. Metrics were measured against
the good copy while the app served the bad one.

All modelling now lives in `forecast_engine.py` and both interfaces import it.
The structural fix matters more than the specific bug — duplicated logic will
diverge eventually.

---

## Approach

**Target.** Absolute room-nights, not occupancy percentage. The two are
informationally identical against a fixed denominator, but counts keep the error
metric interpretable and avoid pathological behaviour on small room types.

**Features.** Current book position at the relevant lead time; same-period-last-year
demand at both 364 days (same weekday) and 365 days (same calendar date); pace
ratios comparing this year's book position against the same point last year;
calendar structure; and a local event calendar with ordinal impact weights.

**Model.** LightGBM per room type and horizon, with monotonic constraints on
demand features and early stopping against a temporal holdout.

**Cancellations and no-shows** are handled implicitly rather than modelled
separately. The target is realised demand — already net of everything that fell
through — so the model learns the historical conversion from books to arrivals
directly. Modelling them explicitly is a reasonable extension but was not
necessary for accuracy.

---

## Running it

```bash
git clone https://github.com/<your-username>/hotel-revenue-forecast.git
cd hotel-revenue-forecast

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data/generate_synthetic_data.py   # build the dataset
python src/models/train_models.py            # train and evaluate
python src/models/predict.py                 # forecast from the sample export

streamlit run src/app/app.py                 # interactive interface
```

Everything runs on generated data — no credentials, no external services.

---

## A note on the data

This was built against a real hotel's PMS extract. That data cannot be
published, and the property is not identified.

The repository therefore ships a **synthetic data generator** that reproduces
the statistical behaviour the pipeline was designed around: monthly and weekday
seasonality, booking curves by lead time, cancellation and no-show rates,
event-driven demand spikes, and year-over-year regime shifts. The property
described in `config/config.yaml` — its name, location, room mix and event
calendar — is fictional.

The generator is not a fig leaf over an empty repo. It encodes a specific model
of how hotel demand behaves, and that model is itself worth reading:
`src/data/generate_synthetic_data.py`.

Results in this README come from the synthetic data, so they are reproducible
by anyone who clones the repo. The findings that motivated the design —
including both bugs above — came from the real deployment.

---

## Layout

```
config/config.yaml          Property definition, strategy, hyperparameters
src/data/                   Synthetic data generator
src/models/
  forecast_engine.py        All modelling logic — single source of truth
  train_models.py           Training and baseline evaluation
  predict.py                Command-line forecasting
src/app/app.py              Streamlit interface
docs/                       Methodology and data notes
```

The pipeline reads every property-specific value from `config/config.yaml`.
Pointing it at a different hotel is a config change, not a code change.

---

## Stack

Python · LightGBM · pandas · NumPy · scikit-learn · Streamlit · Plotly
