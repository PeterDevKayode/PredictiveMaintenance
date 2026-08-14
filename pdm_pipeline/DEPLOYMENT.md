# Deployment Guide

Two things this covers, in order: (1) getting **real XGBoost** numbers for
Chapter 4 — right now the XGBoost rows in your report use a sklearn
GradientBoosting fallback because xgboost couldn't be installed in the
environment that produced them — and (2) putting the dashboard live on
Streamlit Community Cloud, free, so you can screenshot it for Figure 4.1
and demo it in your defense.

Do Part 1 first. The metrics you need for Chapter 4 come out of that step.

---

## Part 1 — Retrain locally with real XGBoost

1. Open a terminal in the `pdm_pipeline` folder.
2. Create a virtual environment and install everything, including real xgboost:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Confirm xgboost actually installed (should NOT print any warning):
   ```bash
   python -c "import xgboost; print('xgboost', xgboost.__version__)"
   ```
4. Retrain on the real full dataset:
   ```bash
   cd src
   python pipeline.py --data ../data/ETTh1.csv --tag ETTh1
   ```
   This overwrites `outputs/models/ETTh1_*.joblib` and
   `outputs/metrics/ETTh1_metrics.json` with results from true XGBoost
   instead of the fallback.
5. Open `outputs/metrics/ETTh1_metrics.json` and **send me its contents**
   (paste it in chat). I'll update Chapter 4's tables and drop the "*
   fallback" footnotes, since those numbers will then be the real thing.

Expect the numbers to move a little from what's in the report now — that's
normal and fine; the discussion in Section 4.8 was written to hold up
regardless of the exact figures, since it's about *why* persistence is a
strong baseline here, not about one specific score.

---

## Part 2 — Deploy the dashboard to Streamlit Community Cloud

This is the free, zero-server-management option. You need a GitHub
account (free) and a Streamlit account (free, sign in with GitHub).

### Step 1 — Push the project to GitHub

```bash
cd pdm_pipeline
git init
git add .
git commit -m "Transformer predictive maintenance pipeline and dashboard"
```
Then create a new empty repository on github.com (no README/license —
you already have files), and push:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

> The `.gitignore` I've added deliberately does NOT exclude
> `data/ETTh1.csv` or `outputs/models/*.joblib` — they get committed on
> purpose, so the deployed app has real data and pretrained models
> immediately, without retraining on every restart.

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Repository: pick the one you just pushed.
4. Branch: `main`.
5. Main file path: `dashboard/app.py`
6. Click **Deploy**.

It builds for a minute or two (installing `requirements.txt`), then gives
you a public URL like `https://<something>.streamlit.app`.

### Step 3 — Use it

In the sidebar: pick `ETTh1.csv`, leave resolution as `h`, and set the
model tag to `ETTh1` (must match what you used in `pipeline.py --tag`).
You'll see the health-state card, RUL estimate, and trend charts.
Screenshot this for Figure 4.1.

### If it fails to deploy

- **"No module named xgboost"** → check `requirements.txt` is at the repo
  root (it is, by default) and lists `xgboost`.
- **"File not found: data/ETTh1.csv"** → check the CSV was actually
  committed (`git status` shouldn't show it as untracked); GitHub has a
  100MB per-file limit, ETTh1.csv is ~2.5MB so this shouldn't trigger.
- **App loads but shows "No trained models found"** → the tag typed in
  the sidebar must exactly match the `--tag` used in `pipeline.py`.

---

## After deployment

Send me the live URL if you'd like — I can't open it myself (no internet
access on my end), but I can help you debug anything that goes wrong
based on the error message, and I'll update Chapter 4 as soon as you
paste the real-XGBoost metrics from Part 1.
