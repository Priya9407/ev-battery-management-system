
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "battery_discharge_features.csv"
MODEL_PATH = BASE_DIR / "capacity_prediction_model.joblib"

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
)


st.markdown(
    """
<style>
/* Main dashboard */
.stApp {
    background-color: #10141f;
    color: #f1f5f9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #202635;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    padding: 18px !important;
    border-radius: 14px !important;
}

/* Metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
}

/* Metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #67e8f9 !important;
}

/* Section headings */
h1, h2, h3 {
    color: #f8fafc !important;
}

/* General text */
.stMarkdown, .stCaption {
    color: #cbd5e1;
}

/* Input labels */
label {
    color: #e2e8f0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------- Load project files --------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_cols"]


st.title("🔋 EV Battery Intelligence")
st.caption(
    "AI-Based EV Battery Management System | "
    "Battery health estimation and charging intelligence"
)

try:
    data = load_data()
    model, feature_columns = load_model()
except Exception as exc:
    st.error(f"Unable to load a project file: {exc}")
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.header("Battery Configuration")

battery_ids = sorted(data["battery_id"].dropna().unique())
selected_battery = st.sidebar.selectbox(
    "Select battery",
    battery_ids,
)

battery_data = data[
    data["battery_id"] == selected_battery
].sort_values("cycle")

selected_row = st.sidebar.slider(
    "Discharge cycle",
    min_value=0,
    max_value=len(battery_data) - 1,
    value=len(battery_data) - 1,
)

row = battery_data.iloc[[selected_row]]
actual_row = battery_data.iloc[selected_row]

# -------------------- Capacity prediction --------------------
feature_columns = [
    "cycle",
    "ambient_temperature_c",
    "Voltage_measured_mean",
    "Voltage_measured_min",
    "Voltage_measured_max",
    "Current_measured_mean",
    "Current_measured_min",
    "Current_measured_max",
    "Temperature_measured_mean",
    "Temperature_measured_min",
    "Temperature_measured_max",
    "Time_mean",
    "Time_min",
    "Time_max",
    "current_abs_mean",
]

missing = [c for c in feature_columns if c not in row.columns]

if missing:
    st.error(f"Missing model input columns: {missing}")
    st.stop()

try:
    predicted_capacity = float(model.predict(row[feature_columns])[0])
except Exception as exc:
    st.error(f"Capacity prediction failed: {exc}")
    st.stop()

rated_capacity = 2.0
predicted_soh = np.clip(
    predicted_capacity / rated_capacity * 100,
    0,
    100,
)

actual_capacity = float(actual_row["capacity_ah"])
actual_soh = float(actual_row["soh"]) * 100

# -------------------- Dashboard metrics --------------------
st.subheader(f"Battery overview — {selected_battery}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "AI-estimated capacity",
    f"{predicted_capacity:.3f} Ah",
)

col2.metric(
    "Estimated state of health",
    f"{predicted_soh:.1f}%",
)

col3.metric(
    "Recorded capacity",
    f"{actual_capacity:.3f} Ah",
)

col4.metric(
    "Cycle number",
    int(actual_row["cycle"]),
)

st.info(
    "Research demonstration: the capacity estimate uses recorded "
    "discharge-cycle measurements. It is not a validated real-time "
    "vehicle battery sensor or a pre-charge prediction."
)

# -------------------- Battery trends --------------------
st.divider()
st.subheader("📈 Battery degradation analysis")

left, right = st.columns(2)

with left:
    st.markdown("**Capacity over discharge cycles**")
    st.line_chart(
        battery_data.set_index("cycle")[["capacity_ah"]]
    )

with right:
    st.markdown("**State of health over discharge cycles**")
    st.line_chart(
        battery_data.set_index("cycle")[["soh"]].mul(100)
    )

# -------------------- Measurement information --------------------
st.divider()
st.subheader("🔬 Selected cycle measurements")

metric_cols = st.columns(3)

measurement_fields = [
    ("Mean voltage", "Voltage_measured_mean", "V"),
    ("Mean current", "Current_measured_mean", "A"),
    ("Mean temperature", "Temperature_measured_mean", "°C"),
]

for col, (label, field, unit) in zip(metric_cols, measurement_fields):
    col.metric(label, f"{float(actual_row[field]):.3f} {unit}")

st.caption(
    "Data source: processed NASA lithium-ion battery cycling dataset. "
    "Capacity and SOH depend on the project's preprocessing assumptions."
)

# -------------------- Dataset preview --------------------
with st.expander("View processed battery dataset"):
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

st.success("Battery dashboard loaded successfully.")


# -------------------- Predict for a New Battery --------------------
st.divider()
st.header("🧪 Predict for a New Battery")
st.write(
    "Upload new discharge-cycle measurements to estimate battery "
    "capacity and state of health using the trained ML model."
)

st.warning(
    "The input must contain the same 15 features used during training. "
    "Predictions for batteries unlike the NASA training batteries may "
    "be inaccurate. This is a research estimate, not a safety assessment."
)

# Downloadable input template
template = pd.DataFrame(columns=feature_columns)
template.loc[0] = [0.0] * len(feature_columns)

st.download_button(
    label="📥 Download CSV input template",
    data=template.to_csv(index=False),
    file_name="new_battery_input_template.csv",
    mime="text/csv",
)

uploaded_csv = st.file_uploader(
    "Upload new battery measurements (CSV)",
    type=["csv"],
    key="new_battery_csv",
)

if uploaded_csv is not None:
    try:
        new_data = pd.read_csv(uploaded_csv)

        missing_columns = [
            column for column in feature_columns if column not in new_data.columns
        ]

        if missing_columns:
            st.error(
                "Your CSV is missing required features: " + ", ".join(missing_columns)
            )
        elif new_data.empty:
            st.error("The uploaded CSV contains no rows.")
        else:
            X_new = new_data[feature_columns].apply(pd.to_numeric, errors="coerce")

            invalid_rows = X_new.isna().all(axis=1)

            if invalid_rows.any():
                st.error(
                    "Some rows contain no usable numeric measurements. "
                    "Please check your CSV."
                )
            else:
                predictions = model.predict(X_new)

                results = new_data.copy()
                results["predicted_capacity_ah"] = predictions
                results["predicted_soh_percent"] = (
                    predictions / rated_capacity * 100
                ).clip(0, 100)

                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "📥 Download prediction results",
                    data=results.to_csv(index=False),
                    file_name="new_battery_predictions.csv",
                    mime="text/csv",
                )

                st.caption(
                    "SOH is estimated as predicted capacity / 2.0 Ah × 100. "
                    "The 2.0 Ah reference is the project's assumed rated capacity."
                )

    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")


# -------------------- Manual Battery Prediction --------------------
st.divider()
st.header("✍️ Predict by Entering Values")
st.write(
    "Enter discharge-cycle measurements below to estimate battery "
    "capacity and state of health."
)

with st.form("manual_battery_prediction"):
    st.subheader("Battery information")

    col1, col2 = st.columns(2)

    with col1:
        cycle = st.number_input(
            "Discharge cycle number",
            min_value=1,
            value=100,
            step=1,
        )
        ambient_temperature_c = st.number_input(
            "Ambient temperature (°C)",
            value=25.0,
            step=0.5,
        )
        voltage_mean = st.number_input(
            "Mean measured voltage (V)",
            value=3.7,
            step=0.01,
        )
        voltage_min = st.number_input(
            "Minimum measured voltage (V)",
            value=3.0,
            step=0.01,
        )
        voltage_max = st.number_input(
            "Maximum measured voltage (V)",
            value=4.2,
            step=0.01,
        )
        current_mean = st.number_input(
            "Mean measured current (A)",
            value=-1.0,
            step=0.01,
        )
        current_min = st.number_input(
            "Minimum measured current (A)",
            value=-2.0,
            step=0.01,
        )
        current_max = st.number_input(
            "Maximum measured current (A)",
            value=0.0,
            step=0.01,
        )

    with col2:
        temperature_mean = st.number_input(
            "Mean battery temperature (°C)",
            value=25.0,
            step=0.5,
        )
        temperature_min = st.number_input(
            "Minimum battery temperature (°C)",
            value=22.0,
            step=0.5,
        )
        temperature_max = st.number_input(
            "Maximum battery temperature (°C)",
            value=30.0,
            step=0.5,
        )
        time_mean = st.number_input(
            "Mean discharge time (seconds)",
            value=1000.0,
            min_value=0.0,
            step=10.0,
        )
        time_min = st.number_input(
            "Minimum discharge time (seconds)",
            value=0.0,
            min_value=0.0,
            step=10.0,
        )
        time_max = st.number_input(
            "Maximum discharge time (seconds)",
            value=2000.0,
            min_value=0.0,
            step=10.0,
        )
        current_abs_mean = st.number_input(
            "Mean absolute current (A)",
            value=1.0,
            min_value=0.0,
            step=0.01,
        )

    predict_clicked = st.form_submit_button(
        "🔋 Predict Battery Capacity",
        type="primary",
        use_container_width=True,
    )

if predict_clicked:
    manual_values = {
        "cycle": cycle,
        "ambient_temperature_c": ambient_temperature_c,
        "Voltage_measured_mean": voltage_mean,
        "Voltage_measured_min": voltage_min,
        "Voltage_measured_max": voltage_max,
        "Current_measured_mean": current_mean,
        "Current_measured_min": current_min,
        "Current_measured_max": current_max,
        "Temperature_measured_mean": temperature_mean,
        "Temperature_measured_min": temperature_min,
        "Temperature_measured_max": temperature_max,
        "Time_mean": time_mean,
        "Time_min": time_min,
        "Time_max": time_max,
        "current_abs_mean": current_abs_mean,
    }

    manual_input = pd.DataFrame(
        [[manual_values[col] for col in feature_columns]],
        columns=feature_columns,
    )

    try:
        manual_capacity = float(model.predict(manual_input)[0])
        manual_soh = manual_capacity / rated_capacity * 100

        st.subheader("🔎 Prediction result")

        result1, result2 = st.columns(2)
        result1.metric(
            "Predicted capacity",
            f"{manual_capacity:.3f} Ah",
        )
        result2.metric(
            "Estimated SOH",
            f"{manual_soh:.1f}%",
        )

        if manual_capacity < 0 or manual_soh > 100:
            st.warning(
                "The model returned a capacity outside the expected "
                "0–2 Ah range. Treat this as an unreliable prediction "
                "and check the input measurements."
            )
        else:
            st.progress(max(0.0, min(1.0, manual_soh / 100)))

        st.caption(
            "Research estimate only. Results depend on input quality "
            "and similarity to the training data. The default values "
            "are examples, not actual measurements."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
