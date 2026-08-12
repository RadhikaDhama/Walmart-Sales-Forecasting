# Demand Forecasting & Inventory Optimization Engine
Access here : https://walmart-sales-forecasting-optvq2vtvh95bdxs9sps3d.streamlit.app/

An end-to-end pipeline that forecasts daily unit demand across Walmart's M5 dataset (30,490 SKU-store series), evaluates forecasts with the competition-correct WRMSSE metric, and converts those forecasts into actionable inventory decisions - safety stock and reorder points at different service levels.

## Problem Statement

Retailers must forecast SKU-level demand across stores to decide how much inventory to hold. Understocking causes stockouts and lost sales; overstocking ties up capital in holding costs and markdowns. This project builds a forecasting pipeline that predicts near-term demand per SKU-store combination, then converts those forecasts into inventory decisions under an explicit cost trade-off - rather than stopping at a forecast accuracy number.

The core challenges that shape the design:
- **Intermittent demand** - many SKUs have long stretches of zero sales, which breaks standard regression metrics like RMSE (a model predicting 0 every day can look artificially accurate).
- **Hierarchical/cross-series structure** - thousands of series with wildly different volume and sparsity profiles.
- **Cost asymmetry** - the cost of holding one extra unit is not the same as the cost of running out; the right amount of safety stock depends on how confident you need to be, not just on the forecast itself.

## Dataset

**[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)** (Walmart, via Kaggle) - 30,490 SKU-store series, ~1,913 days of daily unit sales, across 3 US states (CA, TX, WI), 3 top-level categories (Foods, Household, Hobbies), joined with a calendar table (holidays, SNAP flags) and a weekly sell-price table.

## Pipeline

**1. Data Engineering (PySpark)**  
Raw sales data ships *wide* (one row per SKU-store, ~1,913 day-columns). It's reshaped to *long* format (one row per SKU-store-day — ~58.3M rows), joined with calendar and price data, and checkpointed to Parquet. Run in Spark local mode, since lag/rolling-window features and the wide→long reshape don't scale comfortably in pandas at this row count.

**2. Feature Engineering (PySpark)**  
Lag features (1/7/28-day), rolling mean/std (7/28-day), calendar signals (weekday, month, SNAP, holiday), and a price-drop flag - all computed with `Window.partitionBy(item_id, store_id)` so every series' history stays strictly isolated from every other series'. Per-series **total training-period revenue** (`sales × sell_price`) is also aggregated here in Spark, producing a small lookup table used later for WRMSSE weighting — avoiding a second full-table load in pandas.

**3. Forecasting - Champion/Challenger**  
- **Seasonal-Naive baseline** (forecast = value from 7 days ago) - the floor every real model must beat.
- **SARIMA**, fit on a **stratified 30-series sample** (5 series × 2 volume dimensions × 3 intermittency tiers), diagnosed with ADF stationarity tests, ACF/PACF plots, and validated with a Ljung-Box residual test. Per-series fitting doesn't scale to 30,490 series, so this is a deliberate, stated scoping decision, not a missing result.
- **XGBoost**, trained once as a **global model** across all ~56.6M training rows - shares learned patterns across series (e.g. SNAP-day uplift, price sensitivity), which is what lets it scale to the full dataset and generalize to sparse, low-history series that SARIMA can't fit well.

**4. Evaluation Metric - WRMSSE**  
Standard RMSE fails on intermittent demand. **WRMSSE (Weighted Root Mean Squared Scaled Error)** fixes this two ways: each series' error is *scaled* against that same series' own naive-forecast volatility (so naturally hard-to-forecast series aren't penalized unfairly), and *weighted* by each series' total training-period **dollar revenue** - computed in Spark - so high-revenue SKUs matter more to the aggregate score than low-revenue ones.

**5. Inventory Optimization Layer**  
Converts a forecast + its residual error ($\sigma$) into a **safety stock** and **reorder point**, at 90%/95%/99% service levels:
```
Safety Stock  = Z(service_level) × σ × sqrt(lead_time / horizon)
Reorder Point = expected demand during lead time + Safety Stock
```
This is what turns "you'll probably sell ~X units" into an actual purchasing decision, and makes explicit what extra confidence costs in held inventory.

## Tools

PySpark (distributed data engineering), pandas/NumPy, XGBoost (sklearn API), statsmodels + `pmdarima` (SARIMA, ADF, ACF/PACF, Ljung-Box), scikit-learn, SciPy (`norm.ppf` for service-level Z-scores), Matplotlib, Parquet (intermediate storage), Kaggle Notebooks (execution environment).

## Results

### Table A — Stratified 30-Series Sample (revenue-weighted WRMSSE)

| Stratum | Naive | SARIMA | XGBoost |
|---|---|---|---|
| High_Volume \| Continuous | 0.822 | 0.682 | **0.606** |
| High_Volume \| Intermittent | 1.489 | 1.106 | **1.070** |
| Low_Volume \| Continuous | 1.146 | 0.889 | **0.864** |
| Low_Volume \| Intermittent | 1.703 | 1.234 | **1.208** |
| Med_Volume \| Continuous | 0.886 | **0.614** | 0.629 |
| Med_Volume \| Intermittent | 1.396 | **1.006** | 1.025 |
| **Overall (weighted)** | **1.135** | **0.821** | **0.796** |

XGBoost wins overall and at both volume extremes; SARIMA is competitive - occasionally better - on medium-volume series, where a model fit to a single series' own seasonal structure can out-perform a model generalizing across thousands of others. Intermittency degrades every model's score, confirming sparse demand as the genuinely hard part of the problem.

### Table B — Full-Scale Production Result (all 30,490 series)

| Model | WRMSSE | Improvement vs. Baseline |
|---|---|---|
| Seasonal Naive Baseline | 1.0477 | — |
| **Global XGBoost (Champion)** | **0.7679** | **+26.7%** |

SARIMA is not evaluated at full scale — per-series fitting doesn't scale computationally to 30,490 series; this is a stated scope limitation, not an oversight.

**Inventory example** — `FOODS_3_473 @ WI_3`: 28-day forecast of 73 units, residual $\sigma$ = 2.22. At a 3-day lead time, 95% service level → safety stock of 1.2 units → reorder point of 9.0 units. Low $\sigma$ relative to demand (high-volume, well-forecasted series) means the buffer needed to move from 90% to 99% confidence is small — under one extra unit.

## Conclusion

XGBoost is the deployment champion: it's the only model that scales to the full catalog and it wins overall on the representative sample, delivering a 26.7% WRMSSE improvement over the naive baseline in production. SARIMA's edge on medium-volume, well-behaved series suggests a possible future refinement - a hybrid routing SARIMA to series it wins on and XGBoost elsewhere - though that adds real operational complexity (maintaining two model types in production) that would need to be weighed against a fairly modest accuracy gain.

## Limitations & Scope

- Revenue weighting uses `sales × sell_price` summed over the training period as a WRMSSE weight -
a close proxy for, but not identical to, the exact M5 competition weighting methodology.
- SARIMA benchmarked on 30 stratified series only, by design (see Results).
- Inventory recommendations use XGBoost's forecast/residuals catalog-wide; a more rigorous version would route each series to whichever model performed better on it specifically.
- Lead time (3 days) is a fixed assumption; a production system would source this per-supplier.

## Repository Structure

```
Walmart-Sales-Forecasting/
├── README.md
├── walmart-final.ipynb          # Full pipeline: PySpark → SARIMA/XGBoost → WRMSSE → inventory
├── xgb_model.pkl                # Trained global XGBoost model
└── inventory_summary.csv        # Reorder point recommendations, stratified sample
```
