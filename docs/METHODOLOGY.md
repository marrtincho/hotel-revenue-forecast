# Methodology

Design decisions, and the reasoning behind them. Where a decision was reversed
during development, both the original choice and the reason for changing it are
recorded — the reversals are usually the more informative half.

---

## The forecasting problem

For a given arrival date and room type, predict final realised room-nights,
given the book position at 30, 60 or 90 days before arrival plus calendar and
event context.

Two properties make this harder than a standard tabular regression:

**Partial information at prediction time.** At 90 days out, roughly 13% of final
demand is on the books. The remaining 87% is the thing being predicted. The
signal-to-noise ratio degrades sharply with lead time, which is why a single
model across all horizons underperforms separate models.

**Temporal dependence.** Adjacent dates share demand drivers — a conference
fills Tuesday through Thursday, a festival lifts an entire week. Any evaluation
that splits randomly will leak this structure and report optimistic numbers.

---

## Target variable: counts, not percentages

Occupancy percentage and room count carry identical information when capacity
is fixed. Counts were chosen for three practical reasons.

**Interpretable error.** "The forecast was 8 rooms out" is directly actionable.
"The forecast was 6 percentage points out" requires mental arithmetic against
capacity before it means anything operationally.

**No pathological small-type behaviour.** On a 2-room type, one room is 50% of
capacity. Percentage-based error metrics make such types dominate any average,
even though their absolute contribution to revenue is negligible.

**Direct revenue linkage.** Rooms multiplied by ADR is revenue. No intermediate
conversion step.

Percentages are computed for display only.

---

## Validation: temporal, always

The evaluation split holds out the most recent complete year. Training uses
everything before it.

A random split was never used, and the reason is worth stating explicitly:
booking data is not i.i.d. across rows. Dates within a week share group
bookings, weather, and event exposure. A random split places some of those dates
in training and others in test, letting the model memorise a demand pattern it
would not have access to in production. The resulting metrics can look
excellent while forward performance is poor.

Within training, each room type reserves its most recent 15% of dates as an
internal holdout for early stopping — also temporal, for the same reason.

---

## Baselines

No model ships without clearing both of these.

**Baseline 0 — On the books.** Assume no further bookings. This is the floor:
what the property already knows without any forecasting. It is a weak baseline
by construction, included to quantify how much of the answer pickup actually
represents.

**Baseline 1 — Median historical pickup.** Add the median pickup observed for
that room type, month and lead time. This approximates what an experienced
revenue manager does mentally, and it is a genuinely strong baseline: robust to
outliers, captures seasonality, and requires no infrastructure.

The median was chosen over the mean deliberately. Pickup distributions are
right-skewed — occasional group bookings produce large positive tails — and the
mean is pulled upward by them, producing systematic over-forecasting.

---

## Per-room-type modelling strategy

Room types are not interchangeable. They differ in volume by two orders of
magnitude and in volatility by a factor of three.

| Volume profile | Strategy | Reasoning |
|---|---|---|
| High volume | LightGBM | Enough nightly variation to learn seasonal and event structure |
| Low volume | Median pickup | Insufficient signal; the model finds noise and the baseline is competitive |
| Single room | On the books | Any statistical treatment of a binary outcome is theatre |

This is configured in `config/config.yaml` under `strategy`, not hardcoded.

The decision rule is empirical: a type uses the model only where the model
measurably beats the baseline on held-out data. Where it does not, the
repository says so rather than deploying a model that adds complexity without
accuracy.

---

## Features

**Book position** at the relevant lead time. The primary demand signal.

**Same-period-last-year demand** at two offsets. 364 days lands on the same
weekday, which matters because weekday is among the strongest demand drivers.
365 days lands on the same calendar date, which matters for fixed-date events.
Both are supplied; the model determines which is useful in which context.

**Pace ratios.** This year's book position divided by the same point last year.
This is the feature that lets the model detect a year running structurally ahead
of or behind the previous one, rather than assuming every year repeats.

**Calendar structure.** Day of week, month, quarter, week of year, day of year,
holiday flags, peak-season flag.

**Event calendar.** Local events with ordinal impact weights (none / low /
medium / high), plus days-to-next-event. Events matter most at long lead times,
where the book position carries little information.

### Features deliberately excluded

**OTB revenue and OTB occupancy percentage** were removed. Both are near-perfect
linear transformations of the room count. Their presence split the importance
that should have concentrated on the room count, contributing to the model
becoming insensitive to demand.

**Explicit cancellation and no-show rates** are not features. The target is
realised demand, already net of both, so the historical conversion from books to
arrivals is learned directly. Adding them as separate features would be
double-counting unless the model were restructured to predict gross bookings and
attrition separately — a defensible architecture, but not one that improved
accuracy here.

---

## Monotonic constraints

Three features carry a `+1` monotonic constraint: book position, pace ratio, and
book position relative to last year's realised demand.

The constraint states that increasing the feature can never decrease the
prediction. This encodes a business rule — more rooms sold today cannot mean
fewer rooms sold at arrival — that an unconstrained model is free to violate,
and did.

The constraint costs a small amount of training-set fit and buys two things:
predictions that behave sensibly under sensitivity analysis, and forecasts a
revenue manager can trust to move in the right direction when they refresh the
books.

---

## Sensitivity testing

Aggregate error metrics can hide a model that has stopped using its most
important input. The test that catches this:

1. Take a representative row
2. Hold every feature constant except book position
3. Sweep book position from zero to capacity
4. Plot the prediction

A working model produces a monotonically increasing curve. The failure mode this
caught produced a curve that lurched non-monotonically while aggregate MAE
looked acceptable.

This test is cheap and belongs in any deployment where a model's response to its
primary input is a business requirement rather than an emergent property.

---

## Known limitations

**Event calendar is manually maintained.** Dates shift year to year and new
events appear. A stale calendar degrades long-lead forecasts specifically.

**Three years of history is thin for year-level effects.** The model can detect
a year running ahead or behind via pace ratios, but it cannot anticipate a
regime shift before booking data reveals it.

**No competitor rate or market data.** Forecasts are property-internal. A
competitor opening or closing nearby is invisible until it shows up in bookings.

**No group-booking treatment.** Large group blocks arrive as step changes and
are treated as ordinary demand. A production system would model them separately.

**Long-lead forecasts are structurally uncertain.** At 90 days, most of the
answer has not happened yet. The 90-day forecast is a planning input, not a
pricing instruction.
