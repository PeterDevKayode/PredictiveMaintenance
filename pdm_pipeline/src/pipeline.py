"""
pipeline.py
End-to-end run: load -> clean -> engineer HI/RUL/health_class -> build
features -> chronological split -> train Random Forest / XGBoost
(classification) and SVR / Random Forest / XGBoost (RUL regression) ->
evaluate on the held-out test partition -> save models + metrics.

Usage:
    python pipeline.py --data ../data/ETTh1.csv --source-freq h
    python pipeline.py --data ../data/ETTm1.csv --source-freq 15min
"""
import argparse
import json
import os
import time
import joblib
import numpy as np

from data_loader import load_and_clean
from health_index import engineer_labels
from features import build_feature_table, get_feature_columns
from modeling import (
    chronological_split, make_classifier, make_regressor,
    persistence_baseline_classification, persistence_baseline_regression,
)
from evaluate import classification_report_dict, regression_report_dict, save_metrics, metrics_table

OUT_MODELS = "../outputs/models"
OUT_METRICS = "../outputs/metrics"


def run(data_path: str, source_freq: str = "h", tag: str = None):
    tag = tag or os.path.splitext(os.path.basename(data_path))[0]
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_METRICS, exist_ok=True)

    t0 = time.time()
    print(f"[1/6] Loading and cleaning {data_path} ...")
    df = load_and_clean(data_path, source_freq=source_freq)

    print("[2/6] Engineering Health Index, health-state class and proxy RUL ...")
    df = engineer_labels(df)

    print("[3/6] Building feature table ...")
    df = build_feature_table(df)
    feature_cols = get_feature_columns(df)
    print(f"      {len(df)} rows, {len(feature_cols)} features")

    print("[4/6] Chronological train/val/test split (70/15/15) ...")
    train, val, test = chronological_split(df)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    X_train, X_val, X_test = train[feature_cols], val[feature_cols], test[feature_cols]
    y_cls_train, y_cls_test = train["health_class"].astype(str), test["health_class"].astype(str)
    y_rul_train, y_rul_test = train["RUL"], test["RUL"]

    results = {"classification": {}, "regression": {}}

    # ---------------- Classification: health-state ----------------
    print("[5/6] Training classifiers (Random Forest, XGBoost) ...")
    baseline_pred = persistence_baseline_classification(y_cls_train.iloc[-1], len(y_cls_test))
    results["classification"]["persistence_baseline"] = classification_report_dict(y_cls_test, baseline_pred)

    for kind in ["random_forest", "xgboost"]:
        clf = make_classifier(kind)
        clf.fit(X_train, y_cls_train)
        pred = clf.predict(X_test)
        results["classification"][kind] = classification_report_dict(y_cls_test, pred)
        joblib.dump(clf, f"{OUT_MODELS}/{tag}_{kind}_classifier.joblib")

    # ---------------- Regression: RUL ----------------
    print("[6/6] Training regressors (SVR, Random Forest, XGBoost) ...")
    baseline_pred_r = persistence_baseline_regression(y_rul_train.iloc[-1], len(y_rul_test))
    results["regression"]["persistence_baseline"] = regression_report_dict(y_rul_test, baseline_pred_r)

    for kind in ["svr", "random_forest", "xgboost"]:
        reg = make_regressor(kind)
        reg.fit(X_train, y_rul_train)
        pred = reg.predict(X_test)
        results["regression"][kind] = regression_report_dict(y_rul_test, pred)
        joblib.dump(reg, f"{OUT_MODELS}/{tag}_{kind}_regressor.joblib")

    save_metrics(results, f"{OUT_METRICS}/{tag}_metrics.json")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Metrics saved to {OUT_METRICS}/{tag}_metrics.json")
    print("\n=== Classification (health-state) ===")
    print(metrics_table(results["classification"])[["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]])
    print("\n=== Regression (RUL, hours) ===")
    print(metrics_table(results["regression"])[["rmse_hours", "mae_hours", "r2"]])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/ETTh1.csv", help="Path to ETDataset CSV")
    parser.add_argument("--source-freq", default="h", choices=["h", "15min"],
                         help="Native resolution of the input file (h for ETTh*, 15min for ETTm*)")
    parser.add_argument("--tag", default=None, help="Label used for output filenames")
    args = parser.parse_args()

    run(args.data, source_freq=args.source_freq, tag=args.tag)
