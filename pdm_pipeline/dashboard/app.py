"""
app.py
Lightweight monitoring dashboard for the transformer predictive-maintenance
pipeline (report Chapter 4.7). Shows raw sensor trends, the engineered
Health Index with Healthy/Warning/Critical thresholds, the latest
model-predicted health state, and the model-predicted Remaining Useful
Life (RUL) with a simple maintenance recommendation.

Run with:
    cd dashboard
    streamlit run app.py

Prerequisites:
    pip install streamlit matplotlib
    Trained models must already exist in ../outputs/models/ - produced by
    running `python ../src/pipeline.py --data <path> --tag <tag>` first.
"""
import glob
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_and_clean          # noqa: E402
from health_index import engineer_labels        # noqa: E402
from features import build_feature_table, get_feature_columns  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

RUL_ALERT_HOURS = 24 * 30      # below this -> "schedule inspection within 30 days"
RUL_URGENT_HOURS = 24 * 7      # below this -> urgent

st.set_page_config(page_title="Transformer Predictive Maintenance", layout="wide")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and processing sensor data ...")
def load_dataset(csv_path: str, source_freq: str):
    df = load_and_clean(csv_path, source_freq=source_freq)
    df = engineer_labels(df)
    df = build_feature_table(df)
    return df


KNOWN_MODEL_NAMES = [
    "random_forest_classifier", "xgboost_classifier",
    "svr_regressor", "random_forest_regressor", "xgboost_regressor",
]


@st.cache_resource(show_spinner="Loading trained models ...")
def load_models(tag: str):
    """
    Loads exactly the models trained for this tag. Uses an exact-match
    check against KNOWN_MODEL_NAMES rather than a loose f"{tag}_*.joblib"
    glob, because a loose glob would also match longer tags that happen
    to start with this one (e.g. tag="ETTh1" would otherwise also match
    files saved under tag="ETTh1_partial").
    """
    models = {}
    for name in KNOWN_MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{tag}_{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def status_colour(health_class: str) -> str:
    return {"Healthy": "#2e7d32", "Warning": "#f9a825", "Critical": "#c62828"}.get(health_class, "#616161")


def recommendation(rul_hours: float) -> str:
    if rul_hours <= RUL_URGENT_HOURS:
        return "URGENT: predicted RUL under 1 week - prioritise immediate inspection."
    if rul_hours <= RUL_ALERT_HOURS:
        return "Schedule inspection within 30 days - predicted RUL is trending low."
    return "No immediate action required - continue routine monitoring."


# --------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------- #
st.sidebar.title("Configuration")

available_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not available_csvs:
    st.sidebar.warning("No CSV files found in ../data/. Add an ETDataset CSV "
                        "(e.g. ETTh1.csv) there first.")
    st.stop()

csv_choice = st.sidebar.selectbox("Sensor log file", available_csvs,
                                   format_func=os.path.basename)
source_freq = st.sidebar.selectbox("Native resolution of this file", ["h", "15min"], index=0)
tag = st.sidebar.text_input(
    "Model tag (must match --tag used when running pipeline.py)",
    value=os.path.splitext(os.path.basename(csv_choice))[0],
)

# --------------------------------------------------------------------- #
# Load data + models
# --------------------------------------------------------------------- #
df = load_dataset(csv_choice, source_freq)
models = load_models(tag)

min_date, max_date = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

view = df.loc[str(start_date):str(end_date)]

st.title("Transformer Predictive Maintenance Dashboard")
st.caption(f"Station data: {os.path.basename(csv_choice)}  |  "
           f"{len(view):,} readings shown  |  Tag: {tag}")

if not models:
    st.error(
        f"No trained models found for tag '{tag}' in outputs/models/. "
        f"Run this first:\n\n"
        f"    python ../src/pipeline.py --data {csv_choice} --tag {tag}"
    )
    st.stop()

# --------------------------------------------------------------------- #
# Latest status card
# --------------------------------------------------------------------- #
latest = df.iloc[[-1]]
feature_cols = get_feature_columns(df)

clf_name = "xgboost_classifier" if "xgboost_classifier" in models else next(
    (k for k in models if k.endswith("classifier")), None)
reg_name = "xgboost_regressor" if "xgboost_regressor" in models else next(
    (k for k in models if k.endswith("regressor")), None)

col1, col2, col3 = st.columns(3)

if clf_name:
    pred_class = models[clf_name].predict(latest[feature_cols])[0]
    with col1:
        st.markdown("#### Predicted Health State")
        st.markdown(
            f"<div style='background-color:{status_colour(pred_class)};"
            f"padding:20px;border-radius:8px;text-align:center;'>"
            f"<span style='color:white;font-size:28px;font-weight:bold;'>{pred_class}</span>"
            f"</div>", unsafe_allow_html=True)

if reg_name:
    # Regressors can slightly overshoot the 4,380h training cap (Section
    # 3.6.3 of the report) since nothing constrains their raw output range;
    # clip for display so the dashboard never shows a RUL beyond what the
    # label design considers meaningful.
    pred_rul = min(float(models[reg_name].predict(latest[feature_cols])[0]), 4380.0)
    with col2:
        st.markdown("#### Predicted RUL")
        st.metric(label="Hours remaining (proxy estimate)", value=f"{pred_rul:,.0f} h",
                   delta=f"~{pred_rul/24:,.0f} days")

with col3:
    st.markdown("#### Current Health Index")
    st.metric(label="HI (0-100)", value=f"{latest['HI'].iloc[0]:.1f}")

if reg_name:
    st.info(recommendation(pred_rul))

st.divider()

# --------------------------------------------------------------------- #
# Trend charts
# --------------------------------------------------------------------- #
c1, c2 = st.columns(2)

with c1:
    st.subheader("Oil Temperature")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(view.index, view["OT"], color="#1565c0", linewidth=1)
    ax.set_ylabel("OT (deg C)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with c2:
    st.subheader("Total Load")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(view.index, view["total_load"], color="#6a1b9a", linewidth=1)
    ax.set_ylabel("Total load")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.subheader("Health Index Trend")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(view.index, view["HI"], color="#2e7d32", linewidth=1.2, label="Health Index")
ax.axhline(70, color="#f9a825", linestyle="--", linewidth=1, label="Warning threshold (70)")
ax.axhline(40, color="#c62828", linestyle="--", linewidth=1, label="Critical threshold (40)")
ax.set_ylabel("HI (0-100)")
ax.set_ylim(0, 100)
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
st.pyplot(fig)

with st.expander("Raw feature table (latest 200 rows in range)"):
    st.dataframe(view.tail(200))
