
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
