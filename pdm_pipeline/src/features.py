"""
features.py
Builds the model-ready feature table from the cleaned + labelled dataframe.
All windows are strictly trailing (backward-looking) to avoid leaking
future information into a feature used to predict the present/future.
"""
import numpy as np
import pandas as pd

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
WINDOWS_HOURS = [24, 168, 720]  # 1 day, 1 week, 1 month

# Columns that are labels/targets, not features - excluded from the X matrix.
TARGET_COLS = ["HI", "health_class", "RUL", "F_AA", "rolling_ageing", "load_severity"]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total_load = out[LOAD_COLS].sum(axis=1)
    out["total_load"] = total_load

    for w in WINDOWS_HOURS:
        out[f"OT_roll_mean_{w}h"] = out["OT"].rolling(w, min_periods=1).mean()
        out[f"OT_roll_std_{w}h"] = out["OT"].rolling(w, min_periods=1).std().fillna(0)
        out[f"OT_roll_min_{w}h"] = out["OT"].rolling(w, min_periods=1).min()
        out[f"OT_roll_max_{w}h"] = out["OT"].rolling(w, min_periods=1).max()

        out[f"load_roll_mean_{w}h"] = total_load.rolling(w, min_periods=1).mean()
        out[f"load_roll_std_{w}h"] = total_load.rolling(w, min_periods=1).std().fillna(0)

    return out


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["OT_diff_1h"] = out["OT"].diff(1).fillna(0)
    out["OT_diff_24h"] = out["OT"].diff(24).fillna(0)
    out["load_diff_1h"] = out["total_load"].diff(1).fillna(0)
    out["load_diff_24h"] = out["total_load"].diff(24).fillna(0)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out.index.hour
    month = out.index.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def build_feature_table(df_labelled: pd.DataFrame) -> pd.DataFrame:
    """
    df_labelled: output of health_index.engineer_labels() — must already
    contain OT, the six load columns, HI, health_class and RUL.
    """
    out = add_rolling_features(df_labelled)
    out = add_rate_of_change(out)
    out = add_calendar_features(out)
    # rolling_ageing is excluded from features (it directly determines
    # HI/health_class - see health_index.py); other engineered features
    # below capture equivalent thermal/load signal.
    out = out.dropna()
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in TARGET_COLS]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data_loader import load_and_clean
    from health_index import engineer_labels

    p = sys.argv[1] if len(sys.argv) > 1 else "../data/ETTh1.csv"
    d = load_and_clean(p)
    d = engineer_labels(d)
    d = build_feature_table(d)
    feats = get_feature_columns(d)
    print(f"Feature table shape: {d.shape}")
    print(f"Number of features: {len(feats)}")
    print(feats)
