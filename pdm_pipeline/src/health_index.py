"""
health_index.py
Engineers a physics-informed transformer Health Index (HI), proxy Remaining
Useful Life (RUL), and a discrete health-state label from cleaned oil
temperature (OT) and load data, following the thermal-ageing relationship
used in IEEE C57.91 / IEC 60076-7 transformer loading guides.

This module exists because the ETDataset contains no explicit failure
records — see Chapter 3.5 of the accompanying project report.
"""
import numpy as np
import pandas as pd

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# --- Health Index construction constants (documented, not arbitrary) ---
REF_TEMP_K = 383.0          # 110 degC rated reference hot-spot temperature
AGEING_RATE_CONST = 15000.0  # IEEE C57.91 Arrhenius rate constant
LOAD_WEIGHT = 10.0           # weight of loading-severity penalty in HI
LOAD_WINDOW_HOURS = 168      # 1 week rolling window for loading severity

# Ageing is measured over a TRAILING window (default: 1 year), not as an
# unbounded lifetime cumulative sum. A lifetime cumsum is monotonically
# increasing by construction, which means the most recent (chronologically
# last) portion of any record is always "worst", regardless of how the
# budget is calibrated - this trivially inflates a persistence baseline
# and starves tree models of any test-range values seen in training
# (see report Section 4.8 for a full discussion of this finding). A
# trailing-window ageing measure instead reflects *recent* thermal stress,
# rises and falls with seasonal/loading conditions, and gives train and
# test partitions comparable, learnable value ranges.
AGEING_WINDOW_HOURS = 8760  # 1 year

HI_HEALTHY = 70
HI_WARNING = 40
FAILURE_THRESHOLD = 20       # HI level treated as "end of life" for RUL calc
MAX_RUL_HOURS = 4380         # cap RUL extrapolation at ~6 months
RUL_TREND_WINDOW = 720       # 1-month trailing window used to fit local HI trend


def ageing_acceleration(ot_celsius: pd.Series,
                         ref_k: float = REF_TEMP_K,
                         rate_const: float = AGEING_RATE_CONST) -> pd.Series:
    """
    Per-interval insulation ageing acceleration factor F_AA, from the
    Arrhenius relation in IEEE C57.91: ageing roughly doubles per 6-8 degC
    rise in hot-spot temperature above the rated reference.
    """
    ot_k = ot_celsius + 273.0
    return np.exp(rate_const / ref_k - rate_const / ot_k)


def _normalise(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0 or np.isnan(rng):
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def compute_raw_ageing(df: pd.DataFrame) -> pd.DataFrame:
    """Adds F_AA and rolling_ageing (unnormalised, trailing-window) columns to df."""
    out = df.copy()
    out["F_AA"] = ageing_acceleration(out["OT"])
    out["rolling_ageing"] = out["F_AA"].rolling(AGEING_WINDOW_HOURS, min_periods=24).sum()
    out["rolling_ageing"] = out["rolling_ageing"].fillna(out["F_AA"].expanding(min_periods=1).sum())
    return out


def calibrate_ageing_range(rolling_ageing_train: pd.Series,
                            low_pct: float = 5, high_pct: float = 95) -> tuple:
    """
    Chooses the (low, high) rolling_ageing values that map to HI=100 and
    HI=0 respectively, using percentiles of the TRAINING partition only
    (never validation/test, to avoid leakage across the chronological
    split). Using percentiles rather than raw min/max keeps a handful of
    extreme spikes from compressing the whole normal operating range into
    a narrow band.
    """
    lo = float(np.percentile(rolling_ageing_train.dropna(), low_pct))
    hi = float(np.percentile(rolling_ageing_train.dropna(), high_pct))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def compute_health_index(df: pd.DataFrame, ageing_low: float, ageing_high: float) -> pd.DataFrame:
    """
    Adds load_severity and HI columns to df (df must already have
    rolling_ageing from compute_raw_ageing). ageing_low/high are normally
    produced by calibrate_ageing_range() on the training partition.
    """
    out = df.copy()
    total_load = out[LOAD_COLS].sum(axis=1)
    out["load_severity"] = total_load.rolling(LOAD_WINDOW_HOURS, min_periods=1).mean()

    norm_ageing = ((out["rolling_ageing"] - ageing_low) / (ageing_high - ageing_low)).clip(0, 1)
    norm_load = _normalise(out["load_severity"])  # secondary term; low weight

    out["HI"] = (100 * (1 - norm_ageing) - LOAD_WEIGHT * norm_load).clip(0, 100)
    return out


def add_health_class(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a discrete 3-class health-state label derived from HI."""
    out = df.copy()
    out["health_class"] = pd.cut(
        out["HI"],
        bins=[-0.01, HI_WARNING, HI_HEALTHY, 100.01],
        labels=["Critical", "Warning", "Healthy"],
    )
    return out


def add_proxy_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates proxy RUL(t) in hours: extrapolates the local linear trend of
    HI over a trailing window forward to the point where HI would cross
    FAILURE_THRESHOLD. Capped at MAX_RUL_HOURS. If HI is flat/rising, RUL
    is set to the cap (no foreseeable failure within the horizon).
    """
    out = df.copy()
    hi = out["HI"].values
    n = len(hi)
    rul = np.full(n, float(MAX_RUL_HOURS))

    idx = np.arange(n)
    for i in range(n):
        start = max(0, i - RUL_TREND_WINDOW + 1)
        window_y = hi[start:i + 1]
        window_x = idx[start:i + 1]
        if len(window_y) < 10:
            continue
        # local linear fit of HI vs time-step
        slope, intercept = np.polyfit(window_x, window_y, 1)
        current_hi = hi[i]
        if slope < -1e-9 and current_hi > FAILURE_THRESHOLD:
            steps_to_failure = (FAILURE_THRESHOLD - current_hi) / slope
            rul[i] = float(np.clip(steps_to_failure, 0, MAX_RUL_HOURS))
        elif current_hi <= FAILURE_THRESHOLD:
            rul[i] = 0.0
        # else: flat/improving trend -> stays at cap

    out["RUL"] = rul
    return out


def _train_mask_stratified_by_month(index: pd.DatetimeIndex, train_frac: float) -> np.ndarray:
    """
    Boolean mask selecting the rows that modeling.chronological_split()
    will assign to the training partition (month-stratified - see that
    function's docstring for why). Duplicated here (rather than imported)
    to keep health_index.py free of a dependency on modeling.py.
    """
    mask = np.zeros(len(index), dtype=bool)
    months = index.month
    for m in np.unique(months):
        idxs = np.where(months == m)[0]
        train_end = int(len(idxs) * train_frac)
        mask[idxs[:train_end]] = True
    return mask


def engineer_labels(df: pd.DataFrame, train_frac: float = 0.70) -> pd.DataFrame:
    """
    Full label pipeline: raw rolling ageing -> calibrate ageing_low/high on
    the rows that will fall in the TRAINING partition (month-stratified,
    matching modeling.chronological_split) -> HI -> health_class -> proxy
    RUL, computed over the WHOLE dataframe using that calibration.
    """
    out = compute_raw_ageing(df)
    train_mask = _train_mask_stratified_by_month(out.index, train_frac)
    ageing_low, ageing_high = calibrate_ageing_range(out.loc[train_mask, "rolling_ageing"])
    out = compute_health_index(out, ageing_low=ageing_low, ageing_high=ageing_high)
    out = add_health_class(out)
    out = add_proxy_rul(out)
    out.attrs["ageing_low"] = ageing_low
    out.attrs["ageing_high"] = ageing_high
    return out


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data_loader import load_and_clean

    p = sys.argv[1] if len(sys.argv) > 1 else "../data/ETTh1.csv"
    d = load_and_clean(p)
    d = engineer_labels(d)
    print(d[["OT", "HI", "health_class", "RUL"]].describe(include="all"))
    print("\nClass balance:\n", d["health_class"].value_counts(normalize=True))
