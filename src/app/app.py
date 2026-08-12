"""
app.py
======
Streamlit front end for the forecasting pipeline.

The audience for this interface is a revenue manager, not an analyst. That
shapes several choices: the model is never described in terms of features or
hyperparameters, forecasts are shown next to the current book position so the
number has context, and the app states plainly which room types use the model
and which fall back to a statistical baseline rather than implying uniform
sophistication.

Run with:
    streamlit run src/app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

import forecast_engine as engine  # noqa: E402

st.set_page_config(page_title="Revenue Forecast", page_icon="📈", layout="wide")

SETTINGS = engine.Settings()
ROOM_ORDER = sorted(SETTINGS.inventory, key=lambda rt: -SETTINGS.capacity(rt))

COLOUR_BOOKS = "#94a3b8"
COLOUR_FORECAST = "#2563eb"
COLOUR_ACTUAL = "#16a34a"


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
for key in ("forecast", "snapshot_date"):
    st.session_state.setdefault(key, None)


def load_latest_forecast() -> None:
    """Restore the most recent saved forecast so the app opens with content."""
    if st.session_state["forecast"] is not None:
        return
    saved = sorted(SETTINGS.output_dir.glob("forecast_*.csv"))
    if saved:
        df = pd.read_csv(saved[-1], parse_dates=["stay_date", "snapshot_date"])
        st.session_state["forecast"] = df
        st.session_state["snapshot_date"] = df["snapshot_date"].iloc[0]


load_latest_forecast()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title(SETTINGS.hotel_name)
st.sidebar.caption(f"{SETTINGS.total_rooms} rooms · {len(SETTINGS.inventory)} room types")

page = st.sidebar.radio(
    "Navigation",
    ["Forecast", "Update books", "Model performance"],
    label_visibility="collapsed",
)

st.sidebar.divider()

bundle = engine.load_trained(SETTINGS, log=lambda _: None)
if bundle is not None:
    st.sidebar.caption(
        f"Model trained {pd.Timestamp(bundle['trained_at']):%d %b %Y, %H:%M}"
    )
else:
    st.sidebar.warning("No model trained yet.")

if st.sidebar.button("Retrain model", use_container_width=True):
    log_box = st.sidebar.empty()
    lines: list[str] = []

    def sidebar_log(message: str) -> None:
        lines.append(message)
        log_box.code("\n".join(lines[-5:]))

    with st.spinner("Training..."):
        metrics = engine.train_and_save(SETTINGS, log=sidebar_log)
    st.sidebar.success("Model retrained")
    with st.sidebar.expander("Holdout accuracy"):
        for (room_type, horizon), m in sorted(metrics.items()):
            st.write(
                f"**{room_type}** @ {horizon}d — {m['mae_rooms']:.1f} rooms "
                f"({m['mae_pct_of_capacity']:.1f}% of capacity)"
            )

st.sidebar.divider()
st.sidebar.caption(
    "Demo running on synthetic data. See README for how the dataset is generated."
)


# =============================================================================
# Forecast
# =============================================================================
if page == "Forecast":
    st.title("Occupancy forecast")

    if st.session_state["forecast"] is None:
        st.info("No forecast yet. Go to **Update books** to generate one.")
        st.stop()

    forecast = st.session_state["forecast"]
    snapshot = pd.Timestamp(st.session_state["snapshot_date"])

    st.caption(f"Based on the book position as at {snapshot:%d %B %Y}")

    horizon = st.radio(
        "Lead time", SETTINGS.horizons,
        format_func=lambda h: f"{h} days out", horizontal=True,
    )

    window_start = snapshot + pd.Timedelta(days=horizon)
    window_end = window_start + pd.Timedelta(days=30)
    window = forecast[
        (forecast["horizon_days"] == horizon)
        & (forecast["stay_date"] >= window_start)
        & (forecast["stay_date"] < window_end)
    ]

    if window.empty:
        st.warning("No dates available at this lead time.")
        st.stop()

    st.markdown(f"**Arrivals {window_start:%d %b} — {window_end:%d %b %Y}**")

    summary = window.groupby("room_type_name").agg(
        capacity=("capacity", "first"),
        books=("otb_rooms", "mean"),
        forecast=("predicted_rooms", "mean"),
        method=("method", "first"),
    ).reset_index()
    summary = summary.set_index("room_type_name").reindex(ROOM_ORDER).reset_index()

    total_books = summary["books"].sum()
    total_forecast = summary["forecast"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("On the books", f"{total_books / SETTINGS.total_rooms * 100:.0f}%",
              help="Rooms already reserved for this period.")
    c2.metric("Forecast", f"{total_forecast / SETTINGS.total_rooms * 100:.0f}%",
              delta=f"{(total_forecast - total_books) / SETTINGS.total_rooms * 100:+.0f} pts",
              help="Expected final occupancy once remaining bookings arrive.")
    c3.metric("Expected pickup", f"{total_forecast - total_books:.0f} rooms/night",
              help="Average additional rooms expected per night.")

    st.divider()

    st.subheader("What to look at")
    for insight in engine.generate_insights(forecast, snapshot, SETTINGS):
        {"success": st.success, "warning": st.warning}.get(
            insight["level"], st.info
        )(insight["text"])

    st.divider()

    st.subheader("By room type")

    chart = go.Figure()
    chart.add_bar(
        name="On the books", x=summary["room_type_name"],
        y=summary["books"] / summary["capacity"] * 100, marker_color=COLOUR_BOOKS,
    )
    chart.add_bar(
        name="Forecast", x=summary["room_type_name"],
        y=summary["forecast"] / summary["capacity"] * 100, marker_color=COLOUR_FORECAST,
    )
    chart.update_layout(
        barmode="group", yaxis_title="Occupancy %", height=380,
        legend=dict(orientation="h", y=1.12), margin=dict(t=40, b=20),
    )
    st.plotly_chart(chart, use_container_width=True)

    table = pd.DataFrame({
        "Room type": summary["room_type_name"],
        "Rooms": summary["capacity"],
        "On books": summary["books"].round(1),
        "On books %": (summary["books"] / summary["capacity"] * 100).round(1),
        "Forecast": summary["forecast"].round(1),
        "Forecast %": (summary["forecast"] / summary["capacity"] * 100).round(1),
        "Method": summary["method"],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "Room types with enough booking volume use the trained model. "
        "Smaller types use a median-pickup baseline, which is more reliable "
        "when nightly demand is only a handful of rooms."
    )

    st.divider()

    st.subheader("Night by night")
    selected = st.selectbox("Room type", ROOM_ORDER)
    daily = window[window["room_type_name"] == selected].sort_values("stay_date")

    detail = go.Figure()
    detail.add_scatter(
        x=daily["stay_date"], y=daily["otb_occupancy_pct"],
        name="On the books", mode="lines+markers", line=dict(color=COLOUR_BOOKS),
    )
    detail.add_scatter(
        x=daily["stay_date"], y=daily["predicted_occupancy_pct"],
        name="Forecast", mode="lines+markers", line=dict(color=COLOUR_FORECAST),
    )
    detail.update_layout(
        yaxis_title="Occupancy %", height=340,
        legend=dict(orientation="h", y=1.12), margin=dict(t=40, b=20),
    )
    st.plotly_chart(detail, use_container_width=True)

    st.download_button(
        "Download forecast (CSV)",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{snapshot:%Y%m%d}.csv",
        mime="text/csv",
    )


# =============================================================================
# Update books
# =============================================================================
elif page == "Update books":
    st.title("Update the book position")

    st.markdown(
        "Upload a reservation export from the property management system. "
        "The file should contain one row per room per night, including "
        "forward reservations."
    )

    uploads = st.file_uploader(
        "Reservation export (CSV)", type=["csv"], accept_multiple_files=True
    )

    snapshot_input = st.date_input(
        "Book position date",
        value=pd.Timestamp.today().normalize(),
        help="The date the export was taken. Forecasts are made from this point forward.",
    )

    use_sample = st.checkbox(
        "Use the bundled sample export instead", value=not uploads
    )

    if st.button("Generate forecast", type="primary"):
        if use_sample:
            files = sorted(SETTINGS.raw_pms_dir.glob("*.csv"))
            if not files:
                st.error(
                    "No sample export found. Run "
                    "`python src/data/generate_synthetic_data.py` first."
                )
                st.stop()
        elif uploads:
            SETTINGS.raw_pms_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for upload in uploads:
                target = SETTINGS.raw_pms_dir / upload.name
                target.write_bytes(upload.getbuffer())
                files.append(target)
        else:
            st.error("Upload at least one file, or tick the sample option.")
            st.stop()

        log_box = st.empty()
        lines: list[str] = []

        def app_log(message: str) -> None:
            lines.append(message)
            log_box.code("\n".join(lines[-8:]))

        with st.spinner("Building forecast..."):
            try:
                snapshot = pd.Timestamp(snapshot_input)
                forecast = engine.run_forecast(
                    pms_file_paths=files,
                    snapshot_date=snapshot,
                    settings=SETTINGS,
                    log=app_log,
                )
                if forecast.empty:
                    st.error(
                        "No forecast could be produced. The export may not "
                        "contain reservations beyond the book position date."
                    )
                    st.stop()

                SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
                forecast.to_csv(
                    SETTINGS.output_dir / f"forecast_{snapshot:%Y%m%d}.csv", index=False
                )
                st.session_state["forecast"] = forecast
                st.session_state["snapshot_date"] = snapshot
                st.success("Forecast ready. Open the **Forecast** page to view it.")
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(str(exc))


# =============================================================================
# Model performance
# =============================================================================
elif page == "Model performance":
    st.title("Model performance")

    st.markdown(
        "Forecast accuracy is measured against two reference points a hotel "
        "could use without any model: assuming no further bookings arrive, and "
        "adding the median pickup seen historically for that room type and "
        "month. A model is only worth running if it beats both."
    )

    results_path = SETTINGS.output_dir / "evaluation_results.csv"
    if not results_path.exists():
        st.info(
            "No evaluation results yet. Run "
            "`python src/models/train_models.py` to produce them."
        )
        st.stop()

    results = pd.read_csv(results_path)

    horizon = st.radio(
        "Lead time", sorted(results["horizon"].unique()),
        format_func=lambda h: f"{h} days out", horizontal=True,
    )
    subset = results[results["horizon"] == horizon].sort_values(
        "capacity", ascending=False
    )

    comparison = go.Figure()
    comparison.add_bar(
        name="No further bookings", x=subset["room_type"],
        y=subset["mae_on_books"], marker_color="#cbd5e1",
    )
    comparison.add_bar(
        name="Median pickup", x=subset["room_type"],
        y=subset["mae_median_pickup"], marker_color=COLOUR_BOOKS,
    )
    comparison.add_bar(
        name="Model", x=subset["room_type"],
        y=subset["mae_model"], marker_color=COLOUR_FORECAST,
    )
    comparison.update_layout(
        barmode="group",
        yaxis_title="Mean absolute error (rooms)",
        height=400, legend=dict(orientation="h", y=1.12),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(comparison, use_container_width=True)
    st.caption("Lower is better. Bars show average error in rooms per night.")

    display = pd.DataFrame({
        "Room type": subset["room_type"],
        "Rooms": subset["capacity"],
        "Model error": subset["mae_model"].round(1),
        "Baseline error": subset["mae_median_pickup"].round(1),
        "Improvement": subset["improvement_vs_baseline"].round(1),
        "Method": subset["method"],
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Reading these numbers")
    st.markdown(
        """
Error is reported in rooms per night, not percentage points, because a
percentage is misleading across room types of very different sizes — being
two rooms out matters very differently on a 98-room type than on a 2-room one.

The model does not beat the median-pickup baseline everywhere, and that is
reported rather than hidden. On low-volume room types the baseline is genuinely
competitive: there is not enough signal for a model to learn anything the median
does not already capture. Those types are configured to use the baseline in
production.
        """
    )
