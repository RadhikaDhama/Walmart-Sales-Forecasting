# Demand Forecasting & Inventory Optimization Engine
Access here : https://walmart-sales-forecasting-optvq2vtvh95bdxs9sps3d.streamlit.app/

![PySpark](https://img.shields.io/badge/PySpark-3.4+-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion_Model-2EA043?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **End-to-End Enterprise Supply Chain Pipeline**: Reshapes ~58.3M sales records in PySpark, benchmarks Statistical SARIMA vs Global XGBoost using competition-correct WRMSSE, and converts forecasts into dynamic Safety Stock & Reorder Points under SLA cost trade-offs.

---

## Technical Skills & Core Competencies

| Technical Domain | Skills & Technologies Applied |
|---|---|
| **Distributed Data Engineering** | **PySpark (Local Cluster)** — Reshaped 30,490 series wide→long (~58.3M rows), zero-leakage Window lag/rolling specs, Parquet I/O. |
| **Statistical Time-Series** | **SARIMA (`pmdarima`)** — ADF Stationarity testing, ACF/PACF weekly seasonal ($s=7$) diagnostic order selection, Ljung-Box residual validation. |
| **Machine Learning (GBDT)** | **Global XGBoost ** — Single cross-series model trained on 100% full dataset (55M+ rows), SNAP/price promotion elasticity. |
| **Metric Engineering** | **WRMSSE Metric** — Volatility-scaled & dollar revenue-weighted evaluation (fixing zero-inflated intermittent RMSE bias). |
| **Operations Research** | **Inventory Optimization** — Dynamic Safety Stock & Reorder Point formulation across 90%, 95%, and 99% Service Levels. |
| **Full-Stack Web App** | **Streamlit** — Interactive supply chain application with Plotly scenario planner & purchase order CSV export. |

---

## Executive Summary

- **Problem:** Standard ML pipelines optimize for RMSE, ignoring intermittent zero-sales demand and asymmetrical stockout vs holding costs.
- **Solution:** A 4-stage pipeline that predicts demand uncertainty ($\sigma$) and translates predictions into actionable purchase order triggers.
- **Production Result:** Global XGBoost Champion achieved **0.7679 WRMSSE** across all 30,490 series (**+26.7% error reduction** over baseline).

---

## System Architecture & Pipeline Flow

```
[Raw M5 Wide Data (1,913 Days)]
              │
              ▼
   [PySpark Unpivot & Join] ──► ~58.3M Rows (Calendar, SNAP, Sell Prices)
              │
              ▼
   [Feature Windowing (7/28D Lags)] ──► Parquet Checkpoints (/kaggle/working/)
              │
     ┌────────┴───────────────────────────┐
     ▼                                    ▼
[30-Series Stratified SARIMA]     [Global XGBoost QuantileDMatrix]
 (Unbiased Representative Sample)     (100% Full Catalog - 55M+ Rows)
     │                                    │
     └────────┬───────────────────────────┘
              ▼
     [WRMSSE Evaluation Engine] ──► Table A (Sample) & Table B (Full Dataset)
              │
              ▼
  [Dynamic Inventory Layer] ──► Safety Stock & Reorder Points (90/95/99% SLA)
              │
              ▼
   [Streamlit Web Dashboard] ──► Interactive Purchasing PO Exporter
```

---

## Benchmark Results

### Table A — Stratified 30-Series Head-to-Head (WRMSSE)

| Stratum Category | Naive Baseline | SARIMA (Auto-ARIMA) | Global XGBoost (Champion) |
|---|:---:|:---:|:---:|
| **High_Volume \| Continuous** | 0.8220 | 0.6820 | **0.6060** |
| **High_Volume \| Intermittent** | 1.4890 | 1.1060 | **1.0700** |
| **Low_Volume \| Continuous** | 1.1460 | 0.8890 | **0.8640** |
| **Low_Volume \| Intermittent** | 1.7030 | 1.2340 | **1.2080** |
| **Med_Volume \| Continuous** | 0.8860 | **0.6140** | 0.6290 |
| **Med_Volume \| Intermittent** | 1.3960 | **1.0060** | 1.0250 |
| **OVERALL (Weighted Sample)** | **1.1350** | **0.8210** | **0.7960** |

### Table B — Full-Scale Production Metric (All 30,490 Series)

| Model Architecture | Evaluated Series Scope | WRMSSE Score | Improvement vs. Baseline |
|---|:---:|:---:|:---:|
| **Seasonal Naive Baseline** | 30,490 (Full Catalog) | 1.0477 | Baseline (0.0%) |
| **Global XGBoost Champion** | **30,490 (Full Catalog)** | **0.7679** | **+26.7%** |

---

## Inventory Optimization Formulation

$$\text{Safety Stock} = Z_{\text{service\_level}} \times \sigma \times \sqrt{\frac{\text{Lead Time}}{\text{Horizon}}}$$

$$\text{Reorder Point (ROP)} = (\text{Daily Forecast} \times \text{Lead Time}) + \text{Safety Stock}$$

**Real-World Example (`FOODS_3_473 @ WI_3`):**
- **28-Day Demand Forecast:** 73.0 units | **Residual Volatility ($\sigma$):** 2.22
- **Lead Time:** 3 Days | **Target SLA:** 95% ($Z = 1.64$)
- **Result:** Safety Stock = **1.2 units** $\rightarrow$ Reorder Point = **9.0 units**.

---

<details>
<summary><b>🔍 Deep Dive: Pipeline Execution Phases</b></summary>

### 1. Data & Feature Engineering (PySpark)
- Reshaped 1,913 daily sales columns into long format (~58.3M rows).
- Window functions (`Window.partitionBy("item_id", "store_id").orderBy("date")`) computed 1/7/28-day lags, rolling statistics, and promotional price drop flags without cross-series data leakage.
- Aggregated historical dollar revenue (`sales × sell_price`) per series for WRMSSE weights.

### 2. Unbiased Stratified Sampling
- Sampled 30 representative series across 6 strata (3 Volume Terciles $\times$ 2 Intermittency Tiers split at tier median zero-percentage).
- Eliminated top-volume selection bias (high-volume items are unnaturally smooth and easy to forecast).

### 3. Diagnostics & Model Order Selection
- **ADF Test:** Verified non-stationarity ($p > 0.05$), confirming differencing requirement ($d=1$).
- **ACF/PACF Plots:** Confirmed strong weekly autocorrelation spikes at lags 7, 14, 21, 28, and 35 ($s=7$).
- **Residual Validation:** Ljung-Box test ($p > 0.05$) confirmed white noise residuals.
</details>

<details>
<summary><b>⚠️ Scope & Operational Limitations</b></summary>

- **Revenue Weighting:** Uses `sales × sell_price` summed over historical training period — a close proxy for the official M5 competition weights.
- **SARIMA Benchmarking:** Benchmarked on 30 stratified series by design ($O(N)$ computational runtime does not scale to 30,490 series).
- **Fixed Lead Time:** Assumes 3-day lead time across SKUs (a live ERP integration would dynamically pull lead times per supplier).
</details>

---

## Repository Structure

```
Walmart-Sales-Forecasting/
├── README.md                          # Interactive project documentation
├── app.py                             # Streamlit interactive web dashboard
├── walmart-final.ipynb                # Full executed pipeline notebook
├── xgb_model.pkl                      # Saved production XGBoost champion model
├── inventory_summary.csv              # Reorder point recommendations
└── requirements.txt                   # Deployment dependencies
```
