# Transformer Predictive Maintenance — ML Pipeline

Implements the methodology in Chapter 3 of the accompanying project report:
data cleaning -> physics-informed Health Index / proxy RUL / health-state
labels -> feature engineering -> Random Forest / XGBoost classification ->
SVR / Random Forest / XGBoost RUL regression -> evaluation.

**Want to retrain with real XGBoost and deploy the dashboard live?**
See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full step-by-step guide.

## 1. Setup

```bash
pip install -r requirements.txt
```

(xgboost is optional — if not installed, the pipeline automatically falls
back to sklearn's GradientBoosting models and prints a warning. Install
xgboost for the model actually described in the report.)

## 2. Get the data

Download the ETDataset from https://github.com/zhouhaoyi/ETDataset and
place the CSV(s) in `data/`, e.g. `data/ETTh1.csv`.

## 3. Run

```bash
cd src
python pipeline.py --data ../data/ETTh1.csv --source-freq h --tag ETTh1
```

For the 15-minute resolution files (ETTm1/ETTm2), pass `--source-freq 15min`
— they will be resampled to hourly before the Health Index is computed.

Outputs:
- `outputs/models/<tag>_<model>_classifier.joblib` / `_regressor.joblib`
- `outputs/metrics/<tag>_metrics.json` — accuracy/precision/recall/F1 and
  RMSE/MAE/R² for every model, plus the persistence baseline, matching the
  tables in report Chapter 4.

## 4. Module map

| File | Role |
|---|---|
| `data_loader.py` | Load raw CSV, parse timestamps, impute gaps, winsorise OT |
| `health_index.py` | Physics-informed Health Index, proxy RUL, health-state class (report §3.5) |
| `features.py` | Rolling-window, rate-of-change and calendar features (report §3.6) |
| `modeling.py` | Chronological split + model builders (report §3.7) |
| `evaluate.py` | Classification/regression metrics (report §3.8) |
| `pipeline.py` | Orchestrates the full run end-to-end |
| `dashboard/app.py` | Streamlit monitoring dashboard (report §4.7) |

## 4b. Dashboard

Once you've trained models with `pipeline.py` (so `outputs/models/` is
populated), launch the dashboard:

```bash
pip install streamlit matplotlib
cd dashboard
streamlit run app.py
```

Pick the CSV and matching `--tag` you used when running `pipeline.py` in
the sidebar. It shows: oil-temperature and load trends over a selectable
date range, the Health Index trend with the Warning/Critical thresholds
drawn in, a colour-coded predicted health-state card, and a predicted-RUL
metric with a plain-language maintenance recommendation, all computed
from the same trained models saved by the pipeline (no numbers are
hard-coded).


## 5. Important methodological note — read before quoting results

The ETDataset has **no failure labels**. The Health Index is built from an
IEEE C57.91 thermal-ageing formula, normalised against an ageing "budget"
that is **auto-calibrated on the training partition only** (see
`health_index.calibrate_ageing_budget`) so that the labels contain a
realistic mix of Healthy/Warning/Critical examples instead of staying
pinned at "Healthy" for the whole 2-year window (which is what happens if
you normalise against the literal 20-year design-life budget — real
transformers under normal load shouldn't look critical after 2 years).

Because `cum_ageing` is a monotonically increasing sum, the most recent
(test) portion of any time series will always be "further along" than the
training portion — this is a real characteristic of chronological RUL
prediction, not a bug, and tree models can struggle to extrapolate past
ageing values they never saw in training. If your results show the
classifiers underperforming the persistence baseline on the test set,
that is a genuine, discussable finding (see report §4.8) rather than a
broken pipeline — it demonstrates exactly why the report's future-work
section (§5.5) recommends periodic model retraining / drift monitoring
for real deployment.

## 6. Tested

This pipeline was run end-to-end on a synthetic ETT-format dataset
(17,420 hourly rows) to confirm it executes without errors, produces
non-degenerate class labels, and saves models/metrics correctly. Run it
on the real ETTh1.csv for the numbers to put in your report.
