"""
data_loader.py
Loads and cleans an ETDataset-format CSV (ETTh1.csv / ETTh2.csv / ETTm1.csv / ETTm2.csv).

Expected raw columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
Download the real files from: https://github.com/zhouhaoyi/ETDataset
(place them in ../data/, e.g. data/ETTh1.csv)
"""
import pandas as pd
import numpy as np

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
OT_COL = "OT"
OT_PLAUSIBLE_RANGE = (-20.0, 130.0)


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw ETDataset CSV, parse timestamps, sort chronologically."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}, found: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    missing = [c for c in LOAD_COLS + [OT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {path}")
    return df[LOAD_COLS + [OT_COL]]


def clean(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Fill small gaps, winsorise implausible OT readings, and enforce a
    regular time index at the requested frequency (default hourly).
    """
    df = df.copy()

    # Enforce a regular datetime index so rolling windows are meaningful.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "date"

    # Short gaps (<=3 steps): forward fill. Longer gaps: linear interpolation.
    df = df.ffill(limit=3)
    df = df.interpolate(method="linear", limit_direction="both")

    # Winsorise implausible oil-temperature readings rather than deleting them.
    lo, hi = OT_PLAUSIBLE_RANGE
    df[OT_COL] = df[OT_COL].clip(lower=lo, upper=hi)

    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """For 15-minute (ETTm) data: aggregate to hourly resolution by mean."""
    return df.resample("h").mean()


def load_and_clean(path: str, source_freq: str = "h") -> pd.DataFrame:
    """Convenience wrapper: load -> (optionally resample) -> clean."""
    df = load_raw(path)
    if source_freq != "h":
        df = resample_to_hourly(df)
    df = clean(df, freq="h")
    return df


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "../data/ETTh1.csv"
    d = load_and_clean(p)
    print(d.describe())
    print(f"\nRows: {len(d)}  Range: {d.index.min()} -> {d.index.max()}")
