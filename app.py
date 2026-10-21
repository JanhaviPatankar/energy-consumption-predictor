from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict import (
    get_valid_forecast_origins,
    predict_next_hour,
    get_current_submeter_values,
    load_metadata,
)

from recommendations import generate_recommendations


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_hourly_data():

    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "household_power_hourly.csv"
    )

    df = pd.read_csv(
        file_path,
        parse_dates=["timestamp"]
    )

    return df


@st.cache_data
def load_forecasting_data():

    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "forecasting_features.csv"
    )

    df = pd.read_csv(
        file_path,
        parse_dates=["timestamp"]
    )

    return df


@st.cache_data
def load_prediction_results():

    file_path = (
        PROJECT_ROOT
        / "models"
        / "test_predictions.csv"
    )

    return pd.read_csv(
        file_path,
        parse_dates=["timestamp"]
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚡ AI-Based Household Energy Consumption Predictor'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Predict next-hour household electricity consumption '
    'and get simple energy-saving recommendations.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    hourly_df = load_hourly_data()
    forecasting_df = load_forecasting_data()
    prediction_results = load_prediction_results()
    metadata = load_metadata()

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# ============================================================
# VALID FORECAST ORIGINS
# ============================================================

@st.cache_data
def load_forecast_origins():

    return get_valid_forecast_origins()


forecast_origins = load_forecast_origins()

if not forecast_origins:

    st.error("No valid forecast origins were found.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Prediction Settings")

selected_timestamp = st.sidebar.selectbox(
    "Select forecast origin",
    forecast_origins,
    index=len(forecast_origins) - 1,
    format_func=lambda x: pd.Timestamp(x).strftime(
        "%d %b %Y, %H:%M"
    ),
)


# ============================================================
# COST ESTIMATION
# ============================================================

st.sidebar.subheader("Optional Cost Estimation")

electricity_rate = st.sidebar.number_input(
    "Electricity rate (₹/kWh)",
    min_value=0.0,
    value=0.0,
    step=0.50,
)


# ============================================================
# PEAK PERIOD
# ============================================================

st.sidebar.subheader("Optional Peak Period")

use_peak_period = st.sidebar.checkbox(
    "Use peak-period recommendation"
)

peak_start = None
peak_end = None

if use_peak_period:

    peak_start = st.sidebar.number_input(
        "Peak start hour",
        min_value=0,
        max_value=23,
        value=18,
        step=1,
    )

    peak_end = st.sidebar.number_input(
        "Peak end hour",
        min_value=0,
        max_value=23,
        value=22,
        step=1,
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔮 Predict Next Hour",
    type="primary",
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_kwh, level = predict_next_hour(
            selected_timestamp
        )

        # ----------------------------------------------------
        # Sub-meter values
        # ----------------------------------------------------

        submeters = get_current_submeter_values(
            selected_timestamp
        )

        # ----------------------------------------------------
        # Timestamps
        # ----------------------------------------------------

        forecast_timestamp = pd.Timestamp(
            selected_timestamp
        )

        target_timestamp = (
            forecast_timestamp
            + pd.Timedelta(hours=1)
        )

        target_hour = target_timestamp.hour

        # ----------------------------------------------------
        # Use saved thresholds from metadata
        # ----------------------------------------------------

        thresholds = {
            **metadata["classification_thresholds"],
            **metadata["recommendation_thresholds"],
        }

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        level, recommendations = generate_recommendations(
            predicted_kwh=predicted_kwh,
            current_sub1=submeters["sub1"],
            current_sub2=submeters["sub2"],
            current_sub3=submeters["sub3"],
            thresholds=thresholds,
            peak_start=peak_start,
            peak_end=peak_end,
            forecast_hour=target_hour,
        )


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">🔮 Prediction</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Next-Hour Consumption",
                f"{predicted_kwh:.2f} kWh",
            )

        with col2:

            st.metric(
                "Consumption Level",
                level,
            )

        with col3:

            if electricity_rate > 0:

                estimated_cost = (
                    predicted_kwh * electricity_rate
                )

                st.metric(
                    "Estimated Cost",
                    f"₹{estimated_cost:.2f}",
                )

            else:

                st.metric(
                    "Estimated Cost",
                    "Not configured",
                )


        # ====================================================
        # FORECAST INFORMATION
        # ====================================================

        st.info(
            f"Forecast origin: "
            f"{forecast_timestamp.strftime('%d %B %Y, %H:%M')}  "
            f"| Prediction target: "
            f"{target_timestamp.strftime('%d %B %Y, %H:%M')}"
        )


        # ====================================================
        # ACTUAL VS PREDICTED
        # ====================================================

        selected_row = forecasting_df[
            forecasting_df["timestamp"]
            == forecast_timestamp
        ]

        if not selected_row.empty:

            actual_next_hour = selected_row.iloc[0][
                "next_hour_kwh"
            ]

            if pd.notna(actual_next_hour):

                st.markdown(
                    '<div class="section-title">'
                    '🎯 Prediction vs Actual'
                    '</div>',
                    unsafe_allow_html=True,
                )

                comparison_df = pd.DataFrame(
                    {
                        "Value": [
                            predicted_kwh,
                            float(actual_next_hour),
                        ]
                    },
                    index=[
                        "Predicted",
                        "Actual",
                    ],
                )

                st.bar_chart(
                    comparison_df
                )

                prediction_error = (
                    predicted_kwh
                    - float(actual_next_hour)
                )

                st.caption(
                    f"Prediction error: "
                    f"{prediction_error:+.3f} kWh"
                )


        # ====================================================
        # SUB-METER USAGE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '⚡ Current Sub-Meter Usage'
            '</div>',
            unsafe_allow_html=True,
        )

        sub1_col, sub2_col, sub3_col = st.columns(3)

        with sub1_col:

            st.metric(
                "Sub-meter 1",
                f"{submeters['sub1']:.3f} kWh",
            )

        with sub2_col:

            st.metric(
                "Sub-meter 2",
                f"{submeters['sub2']:.3f} kWh",
            )

        with sub3_col:

            st.metric(
                "Sub-meter 3",
                f"{submeters['sub3']:.3f} kWh",
            )


        # ====================================================
        # ENERGY-SAVING RECOMMENDATIONS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💡 Energy-Saving Recommendations'
            '</div>',
            unsafe_allow_html=True,
        )

        for recommendation in recommendations:

            st.success(
                "💡 " + recommendation
            )


        # ====================================================
        # HISTORICAL CONSUMPTION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Historical Consumption'
            '</div>',
            unsafe_allow_html=True,
        )

        history_df = hourly_df[
            hourly_df["timestamp"]
            <= forecast_timestamp
        ].copy()


        # ----------------------------------------------------
        # Last 24 hours
        # ----------------------------------------------------

        last_24 = history_df.tail(24)

        if not last_24.empty:

            chart_24 = last_24[
                [
                    "timestamp",
                    "active_kwh",
                ]
            ].dropna()

            chart_24 = chart_24.set_index(
                "timestamp"
            )

            st.write("**Last 24 Hours**")

            st.line_chart(
                chart_24
            )


        # ----------------------------------------------------
        # Last 7 days
        # ----------------------------------------------------

        last_7_days = history_df.tail(
            24 * 7
        )

        if not last_7_days.empty:

            chart_7 = last_7_days[
                [
                    "timestamp",
                    "active_kwh",
                ]
            ].dropna()

            chart_7 = chart_7.set_index(
                "timestamp"
            )

            st.write("**Last 7 Days**")

            st.line_chart(
                chart_7
            )


        # ====================================================
        # SUB-METER HISTORY
        # ====================================================


        st.markdown(
            '<div class="section-title">'
            '🔌 Sub-Meter History'
            '</div>',
        unsafe_allow_html=True,
        )

        sub_history = forecasting_df[
            forecasting_df["timestamp"] <= forecast_timestamp
        ].tail(24).copy()

        sub_history = sub_history[
        [
            "timestamp",
            "sub1_kwh_lag_0",
            "sub2_kwh_lag_0",
            "sub3_kwh_lag_0",
        ]
        ]
        sub_history = sub_history.set_index("timestamp")

        sub_history = sub_history.rename(
            columns={
                "sub1_kwh_lag_0": "Sub-meter 1",
                "sub2_kwh_lag_0": "Sub-meter 2",
                "sub3_kwh_lag_0": "Sub-meter 3",
            }
        )

        # Plot each available sub-meter value without
        # requiring all three to be present at every hour.
        st.line_chart(sub_history)


    except Exception as error:

        st.error(
            f"Unable to generate prediction: {error}"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Model Performance'
    '</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

test_metrics = metadata["test_metrics"]

with metric1:

    st.metric(
        "MAE",
        f"{test_metrics['MAE']:.4f} kWh",
    )

with metric2:

    st.metric(
        "RMSE",
        f"{test_metrics['RMSE']:.4f} kWh",
    )

with metric3:

    st.metric(
        "MSE",
        f"{test_metrics['MSE']:.4f}",
    )

with metric4:

    st.metric(
        "R²",
        f"{test_metrics['R2']:.4f}",
    )


st.caption(
    f"Selected model: {metadata['selected_model']}"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Model Comparison'
    '</div>',
    unsafe_allow_html=True,
)

model_results_file = (
    PROJECT_ROOT
    / "models"
    / "model_results.csv"
)

model_results = pd.read_csv(
    model_results_file
)

st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True,
)



# ============================================================
# PREDICTION PERFORMANCE ON TEST DATA
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🎯 Test-Set Predictions'
    '</div>',
    unsafe_allow_html=True,
)

if not prediction_results.empty:

    recent_predictions = prediction_results.tail(168).copy()

    recent_predictions = recent_predictions.set_index(
        "timestamp"
    )

    chart_predictions = recent_predictions[
        [
            "actual_kwh",
            "predicted_kwh",
        ]
    ]

    st.line_chart(
        chart_predictions
    )

    st.caption(
        "Actual vs predicted consumption for the "
        "last 168 test-set hours."
    )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.caption(
    "This application predicts next-hour household electricity "
    "consumption using historical consumption patterns and "
    "provides transparent, rule-based energy-saving suggestions."
)