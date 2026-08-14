"""
modeling.py
Chronological train/val/test split, and model builders for:
  - Health-state classification: Random Forest, XGBoost
  - RUL regression: SVR, Random Forest, XGBoost

If the xgboost package is not installed, XGBoost models fall back to
sklearn's GradientBoosting* implementations so the pipeline still runs
end-to-end (a warning is printed). Install xgboost for the real model
used in the project report: pip install xgboost
"""
import warnings
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn(
        "xgboost not installed - falling back to sklearn GradientBoosting "
        "for XGBoost model slots. Run `pip install xgboost` for the real "
        "model used in the project report."
    )


def chronological_split(df, train_frac=0.70, val_frac=0.15, stratify_by_month=True):
    """
    Split a time-indexed dataframe into train/val/test.

    If stratify_by_month is True (default), the split is done SEPARATELY
    within each calendar month, then recombined - e.g. for every month,
    the chronologically first 70% of that month's hours go to train, the
    next 15% to val, and the last 15% to test. Every split therefore sees
    every season proportionally.

    This matters a great deal for the ETDataset: it spans exactly two
    years (24 months), so a single global 70/15/15 chronological cut puts
    essentially all of val/test in the final ~7 months of the record. For
    ETTh1 that tail happens to be a cooler season, which starves val/test
    of any Warning/Critical examples through pure seasonal coincidence,
    not model quality - a genuine time-series pitfall worth flagging in
    the report (see Chapter 4 discussion). With stratify_by_month=False
    you get the naive single-block split used to first surface this
    problem, kept here for comparison/reproducibility.
    """
    df = df.sort_index()
    if not stratify_by_month:
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    train_parts, val_parts, test_parts = [], [], []
    for _, month_df in df.groupby(df.index.month):
        n = len(month_df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train_parts.append(month_df.iloc[:train_end])
        val_parts.append(month_df.iloc[train_end:val_end])
        test_parts.append(month_df.iloc[val_end:])

    import pandas as pd
    train = pd.concat(train_parts).sort_index()
    val = pd.concat(val_parts).sort_index()
    test = pd.concat(test_parts).sort_index()
    return train, val, test


def make_classifier(kind: str, class_weight="balanced", random_state=42):
    kind = kind.lower()
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight=class_weight, random_state=random_state, n_jobs=-1,
        )
    if kind == "xgboost":
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                reg_lambda=1.0, reg_alpha=0.0, random_state=random_state,
                eval_metric="mlogloss", n_jobs=-1,
            )
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, random_state=random_state,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


def make_regressor(kind: str, random_state=42):
    kind = kind.lower()
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )
    if kind == "xgboost":
        if HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                reg_lambda=1.0, reg_alpha=0.0, random_state=random_state, n_jobs=-1,
            )
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.1, random_state=random_state,
        )
    if kind == "svr":
        # SVR requires scaled features - wrap in a pipeline.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.5)),
        ])
    raise ValueError(f"Unknown regressor kind: {kind}")


def persistence_baseline_classification(y_train_last, n):
    """Naive baseline: predict the most recent known class for every future step."""
    return np.full(n, y_train_last)


def persistence_baseline_regression(y_train_last, n):
    return np.full(n, y_train_last)
