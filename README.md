



<p align="center">
  <a href="https://www.kaggle.com/code/hassanjameelahmed/al-rajhi-bank-xgboost-shap-stock-forecasting" target="_blank">
    <img src="AL Rajhi Bank Stock Forecasting XGBoost.png" alt="AL_Rajhi Bank" width="500">
  </a>
</p>

# 🏦 AL Rajhi Bank XGBoost Stock Forecasting Project
## Technical Specification Document

**Project Code:** ALRAJHI-XGB-2024  
**Target Stock:** 1120.SR (Al Rajhi Banking & Investment Corp)  
**Dataset:** `AL_Rjehei.csv`  
**Notebook:** `AL_Rajhi_XGBoost_Analysis.ipynb`  

---

## 📋 Executive Summary

This document outlines the development of a production-grade XGBoost-based time series forecasting system for predicting Al Rajhi Bank's closing stock prices.  

The project combines:

- Advanced feature engineering  
- Robust cross-validation strategies  
- SHAP-based explainability  

The objective is to deliver actionable trading insights while maintaining full model interpretability for stakeholders.

---

## Phase 1: Problem Definition & Model Selection

### 1.1 Problem Framework

| Element | Specification |
|----------|--------------|
| **Target Variable** | `Close` (Next-day closing price prediction) |
| **Problem Type** | Regression + Time-Series Forecasting |
| **Prediction Horizon** | 1-day ahead (configurable to multi-day) |
| **Data Frequency** | Daily OHLCV (Open, High, Low, Close, Volume) |

---

### 1.2 Algorithm Selection: XGBoost

#### ✅ Justification

- **Non-linearity Handling**  
  Captures complex, non-linear relationships between technical indicators and price movements without explicit kernel engineering.

- **Feature Interaction**  
  Built-in capability to detect interactions between volume spikes and price momentum.

- **Missing Value Robustness**  
  Native handling of missing data via sparsity-aware split finding (critical for technical indicators requiring warm-up periods).

- **Regularization**  
  L1/L2 regularization prevents overfitting to market noise inherent in financial time series.

- **Computational Efficiency**  
  Optimized for tabular data; significantly faster than deep learning alternatives for this dataset size.

---

### ⚠️ Limitations & Mitigations

| Limitation | Impact | Mitigation Strategy |
|------------|--------|---------------------|
| No native sequential awareness | May miss long-term temporal dependencies | Explicit lag features (1, 5, 10, 20 days) + rolling windows |
| Hyperparameter sensitivity | Risk of overfitting to historical patterns | TimeSeriesCV with walk-forward validation |
| Feature engineering requirement | Requires domain expertise for temporal features | Comprehensive technical indicator pipeline (RSI, MACD, Bollinger Bands) |

---

## Phase 2: Feature Engineering (XGBoost Optimized)

### 2.1 Temporal Feature Engineering

#### 📅 Date-Based Features

- `year` — Calendar year (capture macro trends)  
- `month` — Month (1–12), seasonal patterns  
- `day` — Day of month (1–31)  
- `dayofweek` — 0=Monday to 6=Sunday (weekly seasonality)  
- `quarter` — Q1–Q4 (earnings cycle alignment)  
- `is_month_end` — Boolean (portfolio rebalancing effects)  
- `is_quarter_end` — Boolean (institutional reporting impact)  

---

#### ⏳ Lag Features (Temporal Memory)

- `close_lag_1` — Previous day close (t-1)  
- `close_lag_5` — 5-day lag (weekly memory)  
- `close_lag_10` — Bi-weekly pattern  
- `close_lag_20` — Monthly trading memory  
- `volume_lag_1` — Previous volume (liquidity persistence)  

---

#### 📈 Rolling Statistics (Trend & Momentum)

- `ma_5` — 5-day moving average (short-term trend)  
- `ma_20` — 20-day moving average (monthly trend)  
- `ma_ratio` — `ma_5 / ma_20` (golden/death cross indicator)  
- `rolling_std_5` — 5-day volatility  
- `rolling_std_20` — Monthly volatility regime  
- `bb_upper` — Bollinger Band upper (20-day, 2 std)  
- `bb_lower` — Bollinger Band lower (20-day, 2 std)  

---
