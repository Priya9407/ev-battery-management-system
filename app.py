import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# Configuration
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "battery_discharge_features.csv"
MODEL_PATH = BASE_DIR / "capacity_prediction_model.joblib"

RATED_CAPACITY_AH = 2.0  # project's assumed rated capacity
EOL_SOH_PERCENT = 70.0  # common end-of-life threshold
EOL_CAPACITY_AH = RATED_CAPACITY_AH * EOL_SOH_PERCENT / 100

FEATURE_COLUMNS = [
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

FEATURE_LABELS = {
    "cycle": "Discharge cycle",
    "ambient_temperature_c": "Ambient temperature (°C)",
    "Voltage_measured_mean": "Mean voltage (V)",
    "Voltage_measured_min": "Minimum voltage (V)",
    "Voltage_measured_max": "Maximum voltage (V)",
    "Current_measured_mean": "Mean current (A)",
    "Current_measured_min": "Minimum current (A)",
    "Current_measured_max": "Maximum current (A)",
    "Temperature_measured_mean": "Mean battery temperature (°C)",
    "Temperature_measured_min": "Minimum battery temperature (°C)",
    "Temperature_measured_max": "Maximum battery temperature (°C)",
    "Time_mean": "Mean discharge time (s)",
    "Time_min": "Minimum discharge time (s)",
    "Time_max": "Maximum discharge time (s)",
    "current_abs_mean": "Mean absolute current (A)",
}

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Theme and styling
# =====================================================================
def detect_theme():
    """Follow Streamlit's active theme (menu > Settings > Theme, or config.toml)."""
    try:
        theme_type = st.context.theme.type
        if theme_type in ("light", "dark"):
            return theme_type
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "dark"


THEME = detect_theme()

PALETTES = {
    "dark": {
        "BG": "#0b1118",
        "PANEL": "#121b26",
        "PANEL_2": "#172231",
        "BORDER": "#243447",
        "TEXT": "#e6edf5",
        "MUTED": "#8ea3b8",
        "SLATE": "#c3d0de",
        "GRID": "#1d2a3a",
        "LIME": "#9bd66b",
        "AMBER": "#f5b942",
        "CORAL": "#f2676b",
    },
    "light": {
        "BG": "#f5f7fa",
        "PANEL": "#ffffff",
        "PANEL_2": "#eef2f6",
        "BORDER": "#d8e0e8",
        "TEXT": "#14202e",
        "MUTED": "#5b6d80",
        "SLATE": "#334456",
        "GRID": "#e3e9ef",
        "LIME": "#4f9a24",
        "AMBER": "#c98300",
        "CORAL": "#d64550",
    },
}

# Accent colors as (dark theme, light theme)
ACCENTS = {
    "Teal": ("#35d0b5", "#0f9d8a"),
    "Blue": ("#5b9bff", "#2563eb"),
    "Violet": ("#a58bff", "#7c3aed"),
    "Orange": ("#ff9a3c", "#e8710a"),
}

# Sidebar sections are created now so they keep this order on the page.
sb_battery = st.sidebar.container()
sb_appearance = st.sidebar.container()
sb_appearance.divider()
sb_appearance.header("Appearance")
accent_name = sb_appearance.selectbox("Accent color", list(ACCENTS), key="accent")
sb_appearance.caption(
    f"Showing the {THEME} theme. Switch light or dark from the menu (top right) "
    "under Settings, then Theme."
)

_p = PALETTES[THEME]
BG, PANEL, PANEL_2, BORDER = _p["BG"], _p["PANEL"], _p["PANEL_2"], _p["BORDER"]
TEXT, MUTED, SLATE, GRID = _p["TEXT"], _p["MUTED"], _p["SLATE"], _p["GRID"]
LIME, AMBER, CORAL = _p["LIME"], _p["AMBER"], _p["CORAL"]
ACCENT = ACCENTS[accent_name][0 if THEME == "dark" else 1]
TEAL = ACCENT  # historical name: the accent color used across the app
ON_ACCENT = BG if THEME == "dark" else "#ffffff"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] {{
    font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{ background-color: {BG}; color: {TEXT}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {TEXT} !important; }}

/* Headings and text */
h1, h2, h3, h4 {{ color: {TEXT} !important; letter-spacing: -0.01em; }}
label, .stMarkdown p, .stCaption {{ color: {SLATE}; }}
label {{ font-weight: 600 !important; }}

/* Hero */
.hero {{
    display: flex; justify-content: space-between; align-items: flex-end;
    gap: 1rem; flex-wrap: wrap;
    padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, {PANEL_2} 0%, {PANEL} 70%);
    border: 1px solid {BORDER}; border-left: 4px solid {TEAL};
    border-radius: 14px;
}}
.hero-title {{ font-size: 1.9rem; font-weight: 800; color: {TEXT}; letter-spacing: -0.02em; }}
.hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.25rem; }}
.hero-meta {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}

/* Pills */
.pill {{
    display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 700; color: var(--c, {TEXT});
    border: 1px solid var(--c, {BORDER});
    background: color-mix(in srgb, var(--c, {BORDER}) 14%, transparent);
}}

/* KPI cards */
.kpi {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.2rem; height: 100%;
}}
.kpi-label {{ color: {MUTED}; font-size: 0.82rem; font-weight: 600; }}
.kpi-value {{ color: var(--c, {TEXT}); font-size: 1.85rem; font-weight: 800; line-height: 1.25; margin-top: 0.2rem; }}
.kpi-sub {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.15rem; }}

/* Section headings */
.section-title {{ font-size: 1.15rem; font-weight: 700; color: {TEXT}; margin: 0.2rem 0 0.1rem 0; }}
.section-sub {{ color: {MUTED}; font-size: 0.88rem; margin-bottom: 0.8rem; }}

/* Health bar */
.soh-track {{
    position: relative; height: 12px; border-radius: 999px;
    background: {PANEL_2}; border: 1px solid {BORDER}; overflow: visible; margin: 0.6rem 0 0.4rem 0;
}}
.soh-fill {{ height: 100%; border-radius: 999px; }}
.soh-eol {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {CORAL}; }}
.soh-note {{ color: {MUTED}; font-size: 0.78rem; }}

/* Stat rows inside panels */
.stat-row {{
    display: flex; justify-content: space-between; padding: 0.5rem 0;
    border-bottom: 1px solid {BORDER}; font-size: 0.9rem;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-row span:first-child {{ color: {MUTED}; }}
.stat-row span:last-child {{ color: {TEXT}; font-weight: 700; }}

/* Bordered containers act as panels */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {PANEL}; border-color: {BORDER} !important; border-radius: 12px;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs button[data-baseweb="tab"] {{
    padding: 0.7rem 1.1rem; color: {MUTED}; font-weight: 600; background: transparent;
}}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEAL}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {TEAL}; height: 3px; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.4rem; }}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 10px; font-weight: 700; border: 1px solid {BORDER};
}}
[data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {{
    background-color: {TEAL}; color: {ON_ACCENT}; border-color: {TEAL};
}}

/* Inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {{
    background-color: {PANEL_2} !important; border-color: {BORDER} !important;
}}
[data-baseweb="input"] input, [data-baseweb="select"] div {{ color: {TEXT} !important; }}

/* Expanders and forms */
[data-testid="stExpander"], [data-testid="stForm"] {{
    border: 1px solid {BORDER}; border-radius: 12px; background: {PANEL};
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: {PANEL_2}; border: 1px dashed {BORDER}; border-radius: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# Helpers
# =====================================================================
def show_df(df, **kwargs):
    """Render a dataframe full-width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_chart(chart):
    """Render an Altair chart full-width across Streamlit versions."""
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def kpi(label, value, sub="", color=None):
    style = f' style="--c:{color}"' if color else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi"{style}><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def pill(text, color=None):
    style = f' style="--c:{color}"' if color else ""
    return f'<span class="pill"{style}>{text}</span>'


def section(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{title}</div>{sub}', unsafe_allow_html=True
    )


def health_status(soh_percent):
    if soh_percent >= 85:
        return "Excellent", TEAL
    if soh_percent >= 75:
        return "Good", LIME
    if soh_percent >= EOL_SOH_PERCENT:
        return "Aging", AMBER
    return "End of life", CORAL


def soh_bar(soh_percent, color):
    width = float(np.clip(soh_percent, 0, 100))
    return (
        f'<div class="soh-track"><div class="soh-fill" style="width:{width}%;background:{color}"></div>'
        f'<div class="soh-eol" style="left:{EOL_SOH_PERCENT}%"></div></div>'
        f'<div class="soh-note">The red marker is the {EOL_SOH_PERCENT:.0f}% end-of-life threshold.</div>'
    )


def stat_rows(rows):
    html = "".join(
        f"<div class='stat-row'><span>{k}</span><span>{v}</span></div>" for k, v in rows
    )
    st.markdown(html, unsafe_allow_html=True)


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=GRID,
            domainColor=BORDER,
            tickColor=BORDER,
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=SLATE,
            titleColor=MUTED,
            orient="top",
            title=None,
            symbolType="stroke",
        )
    )


def line_chart(df, series, y_title, x="cycle", rule=None, mark_x=None, height=300):
    """series: list of (label, column, color)."""
    long = pd.concat(
        [
            pd.DataFrame({x: df[x], "Series": label, "Value": df[col]})
            for label, col, _ in series
        ],
        ignore_index=True,
    )
    domain = [s[0] for s in series]
    colors = [s[2] for s in series]

    base = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(f"{x}:Q", title="Discharge cycle"),
            y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=colors)),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title="Cycle"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
    )
    layers = [base]
    if rule is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [rule]}))
            .mark_rule(color=CORAL, strokeDash=[6, 4], strokeWidth=1.5)
            .encode(y="y:Q")
        )
    if mark_x is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [mark_x]}))
            .mark_rule(color=MUTED, strokeDash=[2, 3], strokeWidth=1.2)
            .encode(x="x:Q")
        )
    return style_chart(alt.layer(*layers).properties(height=height, width="container"))


# =====================================================================
# Load project files
# =====================================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"]


try:
    data = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load a project file: {exc}")
    st.stop()

required = ["battery_id", "capacity_ah", "soh"] + FEATURE_COLUMNS
missing_required = [c for c in required if c not in data.columns]
if missing_required:
    st.error(f"The dataset is missing required columns: {missing_required}")
    st.stop()


@st.cache_data(show_spinner=False)
def predict_battery(battery_id):
    subset = data[data["battery_id"] == battery_id].sort_values("cycle")
    return model.predict(subset[FEATURE_COLUMNS])


# =====================================================================
# Sidebar
# =====================================================================
sb_battery.header("Battery selection")

battery_ids = sorted(data["battery_id"].dropna().unique())
selected_battery = sb_battery.selectbox("Battery", battery_ids)

battery_data = (
    data[data["battery_id"] == selected_battery]
    .sort_values("cycle")
    .reset_index(drop=True)
)

if len(battery_data) < 2:
    st.warning("This battery has fewer than two recorded cycles.")
    st.stop()

selected_row = sb_battery.slider(
    "Discharge cycle",
    min_value=0,
    max_value=len(battery_data) - 1,
    value=len(battery_data) - 1,
    help="Position in the battery's recorded discharge history.",
)

# =====================================================================
# Predictions for the selected battery
# =====================================================================
try:
    battery_data["pred_capacity"] = predict_battery(selected_battery)
except Exception as exc:
    st.error(f"Capacity prediction failed: {exc}")
    st.stop()

battery_data["pred_soh"] = (
    battery_data["pred_capacity"] / RATED_CAPACITY_AH * 100
).clip(0, 100)
battery_data["soh_percent"] = battery_data["soh"] * 100
battery_data["error_ah"] = battery_data["pred_capacity"] - battery_data["capacity_ah"]

actual_row = battery_data.iloc[selected_row]
predicted_capacity = float(actual_row["pred_capacity"])
predicted_soh = float(actual_row["pred_soh"])
actual_capacity = float(actual_row["capacity_ah"])
actual_soh = float(actual_row["soh_percent"])
selected_cycle = int(actual_row["cycle"])
status_label, status_color = health_status(predicted_soh)

# Model fit on this battery's recorded cycles
mae = float(battery_data["error_ah"].abs().mean())
rmse = float(np.sqrt((battery_data["error_ah"] ** 2).mean()))
ss_res = float((battery_data["error_ah"] ** 2).sum())
ss_tot = float(
    ((battery_data["capacity_ah"] - battery_data["capacity_ah"].mean()) ** 2).sum()
)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

# Rough remaining-life estimate from a linear trend of recorded capacity
cycles_arr = battery_data["cycle"].to_numpy(dtype=float)
cap_arr = battery_data["capacity_ah"].to_numpy(dtype=float)
slope, intercept = np.polyfit(cycles_arr, cap_arr, 1)
fade_per_100 = -slope * 100
if actual_capacity <= EOL_CAPACITY_AH:
    rul_text, rul_sub = "Reached", "Recorded capacity is at or below the threshold"
elif slope < 0:
    eol_cycle = (EOL_CAPACITY_AH - intercept) / slope
    remaining = max(0, int(round(eol_cycle - selected_cycle)))
    rul_text, rul_sub = f"{remaining:,} cycles", "Linear-trend estimate to 70% SOH"
else:
    rul_text, rul_sub = "Not estimable", "Capacity trend is not decreasing"

# Fleet summary
fleet_rows = []
for bid, grp in data.groupby("battery_id"):
    grp = grp.sort_values("cycle")
    last_soh = float(grp["soh"].iloc[-1]) * 100
    fleet_rows.append(
        {
            "Battery": bid,
            "Cycles recorded": len(grp),
            "Initial capacity (Ah)": round(float(grp["capacity_ah"].iloc[0]), 3),
            "Latest capacity (Ah)": round(float(grp["capacity_ah"].iloc[-1]), 3),
            "Latest SOH (%)": round(last_soh, 1),
            "Status": health_status(last_soh)[0],
        }
    )
fleet = pd.DataFrame(fleet_rows)

# Sidebar summary
st.sidebar.divider()
st.sidebar.markdown(
    pill(status_label, status_color) + "&nbsp;" + pill(f"Cycle {selected_cycle}"),
    unsafe_allow_html=True,
)
st.sidebar.caption(
    f"{len(battery_ids)} batteries in the dataset. SOH is capacity divided by "
    f"{RATED_CAPACITY_AH:.1f} Ah. End of life is {EOL_SOH_PERCENT:.0f}% SOH."
)

# =====================================================================
# Header
# =====================================================================
st.markdown(
    f"""
<div class="hero">
<div>
<div class="hero-title">EV Battery Intelligence</div>
<div class="hero-sub">Battery health estimation and charging intelligence from discharge-cycle data</div>
</div>
<div class="hero-meta">{pill(f"Battery {selected_battery}", TEAL)}{pill(f"Cycle {selected_cycle}")}</div>
</div>
""",
    unsafe_allow_html=True,
)

(
    tab_overview,
    tab_degradation,
    tab_cycle,
    tab_fleet,
    tab_batch,
    tab_manual,
    tab_data,
    tab_about,
) = st.tabs(
    [
        "Overview",
        "Degradation",
        "Cycle inspector",
        "Fleet comparison",
        "Batch prediction",
        "Manual prediction",
        "Data",
        "About",
    ]
)

# =====================================================================
# Tab 1: Overview
# =====================================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi(
            "AI-estimated capacity",
            f"{predicted_capacity:.3f} Ah",
            f"Rated {RATED_CAPACITY_AH:.1f} Ah",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi(
            "Estimated state of health",
            f"{predicted_soh:.1f}%",
            status_label,
            status_color,
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi(
            "Recorded capacity",
            f"{actual_capacity:.3f} Ah",
            f"Model difference {predicted_capacity - actual_capacity:+.3f} Ah",
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        kpi(
            "Cycle number",
            f"{selected_cycle}",
            f"{selected_row + 1} of {len(battery_data)} recorded cycles",
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        with st.container(border=True):
            section(
                "Capacity: recorded and estimated",
                "The dashed grey line marks the selected cycle.",
            )
            show_chart(
                line_chart(
                    battery_data,
                    [
                        ("Recorded", "capacity_ah", SLATE),
                        ("AI-estimated", "pred_capacity", TEAL),
                    ],
                    "Capacity (Ah)",
                    rule=EOL_CAPACITY_AH,
                    mark_x=selected_cycle,
                )
            )

    with right:
        with st.container(border=True):
            section("Health summary")
            st.markdown(
                f'<div class="kpi-value" style="--c:{status_color}">{predicted_soh:.1f}% '
                f"<span style='font-size:1rem;font-weight:700'>{status_label}</span></div>"
                + soh_bar(predicted_soh, status_color),
                unsafe_allow_html=True,
            )
            st.write("")
            stat_rows(
                [
                    ("Recorded SOH", f"{actual_soh:.1f}%"),
                    ("Estimated remaining life", rul_text),
                    ("Capacity fade", f"{fade_per_100:.3f} Ah per 100 cycles"),
                    ("Mean absolute error", f"{mae:.3f} Ah"),
                    ("R² on this battery", f"{r2:.3f}" if not np.isnan(r2) else "n/a"),
                ]
            )
            st.caption(rul_sub)

    st.info(
        "Research demonstration: the capacity estimate uses recorded discharge-cycle "
        "measurements. It is not a validated real-time vehicle battery sensor or a "
        "pre-charge prediction. Error figures are computed on this battery's recorded "
        "cycles and are not held-out accuracy if the battery was part of the training data."
    )

# =====================================================================
# Tab 2: Degradation
# =====================================================================
with tab_degradation:
    section(
        "Battery degradation analysis",
        "Compare recorded values with the model's estimates across the discharge history.",
    )

    cmin, cmax = int(battery_data["cycle"].min()), int(battery_data["cycle"].max())
    if cmin < cmax:
        cycle_range = st.slider(
            "Cycle range", cmin, cmax, (cmin, cmax), key="deg_range"
        )
    else:
        cycle_range = (cmin, cmax)
    view = battery_data[battery_data["cycle"].between(*cycle_range)]

    show_estimates = st.toggle("Show AI estimates", value=True)

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            section("Capacity over discharge cycles")
            series = [("Recorded", "capacity_ah", SLATE)]
            if show_estimates:
                series.append(("AI-estimated", "pred_capacity", TEAL))
            show_chart(
                line_chart(
                    view,
                    series,
                    "Capacity (Ah)",
                    rule=EOL_CAPACITY_AH,
                    mark_x=selected_cycle,
                )
            )
    with b:
        with st.container(border=True):
            section("State of health over discharge cycles")
            series = [("Recorded", "soh_percent", SLATE)]
            if show_estimates:
                series.append(("AI-estimated", "pred_soh", TEAL))
            show_chart(
                line_chart(
                    view,
                    series,
                    "State of health (%)",
                    rule=EOL_SOH_PERCENT,
                    mark_x=selected_cycle,
                )
            )

    with st.container(border=True):
        section(
            "Estimation error",
            "Estimated minus recorded capacity. Values near zero mean the model tracks the recorded data closely.",
        )
        show_chart(
            line_chart(
                view.assign(zero=0.0),
                [("Error (Ah)", "error_ah", AMBER)],
                "Error (Ah)",
                rule=0.0,
                mark_x=selected_cycle,
                height=220,
            )
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Mean absolute error", f"{mae:.3f} Ah"), unsafe_allow_html=True)
    m2.markdown(kpi("Root mean square error", f"{rmse:.3f} Ah"), unsafe_allow_html=True)
    m3.markdown(
        kpi("Largest error", f"{battery_data['error_ah'].abs().max():.3f} Ah"),
        unsafe_allow_html=True,
    )
    m4.markdown(
        kpi("Capacity fade", f"{fade_per_100:.3f} Ah", "Per 100 cycles, linear fit"),
        unsafe_allow_html=True,
    )

# =====================================================================
# Tab 3: Cycle inspector
# =====================================================================
with tab_cycle:
    section(
        f"Measurements for cycle {selected_cycle}",
        "Summary statistics of the selected discharge cycle, compared with the battery's first recorded cycle.",
    )

    first_row = battery_data.iloc[0]
    headline = [
        ("Mean voltage", "Voltage_measured_mean", "V"),
        ("Mean current", "Current_measured_mean", "A"),
        ("Mean temperature", "Temperature_measured_mean", "°C"),
        ("Mean discharge time", "Time_mean", "s"),
    ]
    cols = st.columns(4)
    for col, (label, field, unit) in zip(cols, headline):
        value = float(actual_row[field])
        delta = value - float(first_row[field])
        col.markdown(
            kpi(label, f"{value:.3f} {unit}", f"{delta:+.3f} {unit} since first cycle"),
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            section("All model inputs")
            stat_rows(
                [
                    (FEATURE_LABELS[c], f"{float(actual_row[c]):.3f}")
                    for c in FEATURE_COLUMNS
                ]
            )

    with right:
        with st.container(border=True):
            section("Feature trend")
            feature = st.selectbox(
                "Measurement",
                FEATURE_COLUMNS[1:],
                format_func=lambda c: FEATURE_LABELS[c],
                key="trend_feature",
            )
            show_chart(
                line_chart(
                    battery_data,
                    [(FEATURE_LABELS[feature], feature, TEAL)],
                    FEATURE_LABELS[feature],
                    mark_x=selected_cycle,
                    height=340,
                )
            )

# =====================================================================
# Tab 4: Fleet comparison
# =====================================================================
with tab_fleet:
    section(
        "Fleet comparison",
        "Capacity fade for every battery in the dataset. The selected battery is highlighted.",
    )

    sel_py = (
        selected_battery.item()
        if hasattr(selected_battery, "item")
        else selected_battery
    )
    fleet_chart = (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("cycle:Q", title="Discharge cycle"),
            y=alt.Y(
                "capacity_ah:Q", title="Capacity (Ah)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "battery_id:N", title="Battery", scale=alt.Scale(scheme="tableau10")
            ),
            opacity=alt.condition(
                f"datum.battery_id == {json.dumps(sel_py)}",
                alt.value(1.0),
                alt.value(0.3),
            ),
            tooltip=[
                alt.Tooltip("battery_id:N", title="Battery"),
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("capacity_ah:Q", title="Capacity (Ah)", format=".3f"),
            ],
        )
    )
    eol_rule = (
        alt.Chart(pd.DataFrame({"y": [EOL_CAPACITY_AH]}))
        .mark_rule(color=CORAL, strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    with st.container(border=True):
        show_chart(
            style_chart(
                alt.layer(fleet_chart, eol_rule).properties(
                    height=360, width="container"
                )
            )
        )

    f1, f2, f3 = st.columns(3)
    f1.markdown(kpi("Batteries", f"{len(fleet)}"), unsafe_allow_html=True)
    f2.markdown(
        kpi("Average latest SOH", f"{fleet['Latest SOH (%)'].mean():.1f}%"),
        unsafe_allow_html=True,
    )
    f3.markdown(
        kpi(
            "At or below end of life",
            f"{int((fleet['Latest SOH (%)'] < EOL_SOH_PERCENT).sum())}",
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    with st.container(border=True):
        section("Battery summary")
        show_df(fleet)

# =====================================================================
# Tab 5: Batch prediction
# =====================================================================
with tab_batch:
    section(
        "Predict for a new battery",
        "Upload discharge-cycle measurements to estimate capacity and state of health with the trained model.",
    )

    st.warning(
        f"The CSV must contain the same {len(FEATURE_COLUMNS)} features used during training. "
        "Predictions for batteries unlike the NASA training batteries may be inaccurate. "
        "This is a research estimate, not a safety assessment."
    )

    template = pd.DataFrame(columns=FEATURE_COLUMNS)
    template.loc[0] = [0.0] * len(FEATURE_COLUMNS)

    u1, u2 = st.columns([2, 1])
    with u2:
        with st.container(border=True):
            section("Input template")
            st.caption("Start from a CSV with the required column names.")
            st.download_button(
                "Download CSV template",
                data=template.to_csv(index=False),
                file_name="new_battery_input_template.csv",
                mime="text/csv",
            )
            with st.expander("Required columns"):
                st.code("\n".join(FEATURE_COLUMNS), language="text")

    with u1:
        with st.container(border=True):
            section("Upload measurements")
            uploaded_csv = st.file_uploader(
                "Battery measurements (CSV)",
                type=["csv"],
                key="new_battery_csv",
            )

    if uploaded_csv is not None:
        try:
            new_data = pd.read_csv(uploaded_csv)
            missing_columns = [c for c in FEATURE_COLUMNS if c not in new_data.columns]

            if missing_columns:
                st.error(
                    "Your CSV is missing required features: "
                    + ", ".join(missing_columns)
                )
            elif new_data.empty:
                st.error("The uploaded CSV contains no rows.")
            else:
                X_new = new_data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
                bad_rows = X_new.isna().any(axis=1)

                if bad_rows.any():
                    st.error(
                        f"{int(bad_rows.sum())} row(s) contain missing or non-numeric values "
                        f"(first affected row: {int(bad_rows.idxmax()) + 2} in the file). "
                        "Fix these rows and upload again."
                    )
                else:
                    predictions = model.predict(X_new)
                    results = new_data.copy()
                    results["predicted_capacity_ah"] = predictions
                    results["predicted_soh_percent"] = (
                        predictions / RATED_CAPACITY_AH * 100
                    ).clip(0, 100)

                    st.write("")
                    section("Prediction results")

                    r1, r2c, r3, r4 = st.columns(4)
                    r1.markdown(
                        kpi("Rows predicted", f"{len(results)}"), unsafe_allow_html=True
                    )
                    r2c.markdown(
                        kpi(
                            "Mean capacity",
                            f"{results['predicted_capacity_ah'].mean():.3f} Ah",
                        ),
                        unsafe_allow_html=True,
                    )
                    r3.markdown(
                        kpi(
                            "Mean SOH",
                            f"{results['predicted_soh_percent'].mean():.1f}%",
                        ),
                        unsafe_allow_html=True,
                    )
                    r4.markdown(
                        kpi(
                            "Lowest SOH",
                            f"{results['predicted_soh_percent'].min():.1f}%",
                        ),
                        unsafe_allow_html=True,
                    )

                    out_of_range = int(
                        (
                            (predictions < 0) | (predictions > RATED_CAPACITY_AH * 1.05)
                        ).sum()
                    )
                    if out_of_range:
                        st.warning(
                            f"{out_of_range} prediction(s) fall outside the expected "
                            f"0 to {RATED_CAPACITY_AH:.1f} Ah range. Treat them as unreliable."
                        )

                    st.write("")
                    with st.container(border=True):
                        section("Predicted state of health")
                        show_chart(
                            line_chart(
                                results.sort_values("cycle"),
                                [("Predicted SOH", "predicted_soh_percent", TEAL)],
                                "State of health (%)",
                                rule=EOL_SOH_PERCENT,
                                height=260,
                            )
                        )

                    with st.container(border=True):
                        show_df(results)
                        st.download_button(
                            "Download prediction results",
                            data=results.to_csv(index=False),
                            file_name="new_battery_predictions.csv",
                            mime="text/csv",
                        )

                    st.caption(
                        f"SOH is estimated as predicted capacity / {RATED_CAPACITY_AH:.1f} Ah × 100. "
                        "The reference is the project's assumed rated capacity."
                    )
        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

# =====================================================================
# Tab 6: Manual prediction
# =====================================================================
with tab_manual:
    section(
        "Predict by entering values",
        "Enter discharge-cycle measurements to estimate battery capacity and state of health.",
    )

    use_selected = st.checkbox(
        f"Start from battery {selected_battery}, cycle {selected_cycle}",
        value=False,
        help="Fills the form with the measurements of the cycle selected in the sidebar.",
    )

    if use_selected:
        defaults = {c: float(actual_row[c]) for c in FEATURE_COLUMNS}
        defaults["cycle"] = max(1, int(actual_row["cycle"]))
        for c in ("Time_mean", "Time_min", "Time_max", "current_abs_mean"):
            defaults[c] = max(0.0, defaults[c])
    else:
        defaults = {
            "cycle": 100,
            "ambient_temperature_c": 25.0,
            "Voltage_measured_mean": 3.7,
            "Voltage_measured_min": 3.0,
            "Voltage_measured_max": 4.2,
            "Current_measured_mean": -1.0,
            "Current_measured_min": -2.0,
            "Current_measured_max": 0.0,
            "Temperature_measured_mean": 25.0,
            "Temperature_measured_min": 22.0,
            "Temperature_measured_max": 30.0,
            "Time_mean": 1000.0,
            "Time_min": 0.0,
            "Time_max": 2000.0,
            "current_abs_mean": 1.0,
        }

    with st.form("manual_battery_prediction"):
        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown("**Operating conditions**")
            cycle = st.number_input(
                FEATURE_LABELS["cycle"],
                min_value=1,
                value=int(defaults["cycle"]),
                step=1,
            )
            ambient_temperature_c = st.number_input(
                FEATURE_LABELS["ambient_temperature_c"],
                value=defaults["ambient_temperature_c"],
                step=0.5,
            )
            time_mean = st.number_input(
                FEATURE_LABELS["Time_mean"],
                value=defaults["Time_mean"],
                min_value=0.0,
                step=10.0,
            )
            time_min = st.number_input(
                FEATURE_LABELS["Time_min"],
                value=defaults["Time_min"],
                min_value=0.0,
                step=10.0,
            )
            time_max = st.number_input(
                FEATURE_LABELS["Time_max"],
                value=defaults["Time_max"],
                min_value=0.0,
                step=10.0,
            )

        with g2:
            st.markdown("**Voltage and current**")
            voltage_mean = st.number_input(
                FEATURE_LABELS["Voltage_measured_mean"],
                value=defaults["Voltage_measured_mean"],
                step=0.01,
            )
            voltage_min = st.number_input(
                FEATURE_LABELS["Voltage_measured_min"],
                value=defaults["Voltage_measured_min"],
                step=0.01,
            )
            voltage_max = st.number_input(
                FEATURE_LABELS["Voltage_measured_max"],
                value=defaults["Voltage_measured_max"],
                step=0.01,
            )
            current_mean = st.number_input(
                FEATURE_LABELS["Current_measured_mean"],
                value=defaults["Current_measured_mean"],
                step=0.01,
            )
            current_min = st.number_input(
                FEATURE_LABELS["Current_measured_min"],
                value=defaults["Current_measured_min"],
                step=0.01,
            )
            current_max = st.number_input(
                FEATURE_LABELS["Current_measured_max"],
                value=defaults["Current_measured_max"],
                step=0.01,
            )
            current_abs_mean = st.number_input(
                FEATURE_LABELS["current_abs_mean"],
                value=defaults["current_abs_mean"],
                min_value=0.0,
                step=0.01,
            )

        with g3:
            st.markdown("**Battery temperature**")
            temperature_mean = st.number_input(
                FEATURE_LABELS["Temperature_measured_mean"],
                value=defaults["Temperature_measured_mean"],
                step=0.5,
            )
            temperature_min = st.number_input(
                FEATURE_LABELS["Temperature_measured_min"],
                value=defaults["Temperature_measured_min"],
                step=0.5,
            )
            temperature_max = st.number_input(
                FEATURE_LABELS["Temperature_measured_max"],
                value=defaults["Temperature_measured_max"],
                step=0.5,
            )

        try:
            predict_clicked = st.form_submit_button(
                "Predict battery capacity", type="primary", width="stretch"
            )
        except Exception:
            predict_clicked = st.form_submit_button(
                "Predict battery capacity", type="primary", use_container_width=True
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

        # Input sanity checks
        input_issues = []
        for label, lo, mid, hi in [
            ("Voltage", voltage_min, voltage_mean, voltage_max),
            ("Current", current_min, current_mean, current_max),
            ("Battery temperature", temperature_min, temperature_mean, temperature_max),
            ("Discharge time", time_min, time_mean, time_max),
        ]:
            if not (lo <= mid <= hi):
                input_issues.append(
                    f"{label}: the minimum, mean and maximum values are not in order."
                )

        manual_input = pd.DataFrame(
            [[manual_values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
        )

        try:
            manual_capacity = float(model.predict(manual_input)[0])
            manual_soh = manual_capacity / RATED_CAPACITY_AH * 100
            m_label, m_color = health_status(max(0.0, min(100.0, manual_soh)))

            st.write("")
            section("Prediction result")

            res1, res2, res3 = st.columns(3)
            res1.markdown(
                kpi("Predicted capacity", f"{manual_capacity:.3f} Ah"),
                unsafe_allow_html=True,
            )
            res2.markdown(
                kpi("Estimated SOH", f"{manual_soh:.1f}%", color=m_color),
                unsafe_allow_html=True,
            )
            res3.markdown(
                kpi("Health status", m_label, color=m_color), unsafe_allow_html=True
            )

            if manual_capacity < 0 or manual_soh > 100:
                st.warning(
                    f"The model returned a capacity outside the expected 0 to {RATED_CAPACITY_AH:.1f} Ah "
                    "range. Treat this as an unreliable prediction and check the input measurements."
                )
            else:
                st.markdown(soh_bar(manual_soh, m_color), unsafe_allow_html=True)

            for issue in input_issues:
                st.warning(issue)

            st.caption(
                "Research estimate only. Results depend on input quality and similarity to the "
                "training data. The default values are examples, not actual measurements."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# =====================================================================
# Tab 7: Data
# =====================================================================
with tab_data:
    section(
        "Processed battery dataset",
        "Browse, filter and export the data behind the dashboard.",
    )

    scope = st.radio(
        "Show",
        [f"Battery {selected_battery}", "All batteries"],
        horizontal=True,
    )
    table = battery_data[list(data.columns)] if scope != "All batteries" else data

    d1, d2, d3 = st.columns(3)
    d1.markdown(kpi("Rows", f"{len(table):,}"), unsafe_allow_html=True)
    d2.markdown(kpi("Columns", f"{table.shape[1]}"), unsafe_allow_html=True)
    d3.markdown(
        kpi(
            "Cycle range", f"{int(table['cycle'].min())} to {int(table['cycle'].max())}"
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    with st.container(border=True):
        show_df(table)
        st.download_button(
            "Download table as CSV",
            data=table.to_csv(index=False),
            file_name="battery_data_export.csv",
            mime="text/csv",
        )

    with st.expander("Summary statistics"):
        st.dataframe(table.describe().T, use_container_width=True)

# =====================================================================
# Tab 8: About
# =====================================================================
with tab_about:
    section("About this dashboard")
    a1, a2 = st.columns(2)

    with a1:
        with st.container(border=True):
            st.markdown("**What it does**")
            st.markdown(
                "- Estimates discharge capacity from cycle-level voltage, current, temperature and time statistics.\n"
                "- Converts capacity to state of health (SOH) against a rated capacity.\n"
                "- Tracks degradation across the full discharge history of each battery.\n"
                "- Compares batteries and predicts for new uploaded or manually entered measurements."
            )
        with st.container(border=True):
            st.markdown("**Data source**")
            st.markdown(
                "Processed NASA lithium-ion battery cycling dataset. Capacity and SOH depend on "
                "the project's preprocessing assumptions."
            )

    with a2:
        with st.container(border=True):
            st.markdown("**Assumptions**")
            st.markdown(
                f"- Rated capacity: {RATED_CAPACITY_AH:.1f} Ah\n"
                f"- End-of-life threshold: {EOL_SOH_PERCENT:.0f}% SOH ({EOL_CAPACITY_AH:.2f} Ah)\n"
                "- Remaining life is a linear extrapolation of recorded capacity and is only a rough guide."
            )
        with st.container(border=True):
            st.markdown("**Limitations**")
            st.markdown(
                "- Research demonstration, not a validated real-time vehicle sensor.\n"
                "- Not a pre-charge prediction and not a safety assessment.\n"
                "- Accuracy drops for batteries unlike the NASA training batteries.\n"
                "- Error metrics are not held-out accuracy when the battery was in the training data."
            )
