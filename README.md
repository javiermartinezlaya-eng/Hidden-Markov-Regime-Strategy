# Hidden-Markov-Regime-Strategy

Quantitative trading framework based on **Hidden Markov Models (HMM)** for market regime identification, with an emphasis on **out-of-sample validation**, **robustness testing**, and **risk-adjusted performance**.

This project implements a full research pipeline including:

- Walk-forward training (no look-ahead bias)
- Block bootstrap Monte Carlo inference
- Deflated Sharpe Ratio (DSR) to account for data-snooping
- Volatility-matched benchmarking against Buy & Hold

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/d4950d4f-5284-4da4-ab4b-56c64b7c9c0c" />

The objective is not to maximize raw returns, but to construct a **statistically defensible strategy with controlled downside risk**.

---

## Overview

Financial markets exhibit regime-dependent behavior, where risk and return characteristics change over time.  
This project builds a **regime-aware trading strategy on QQQ** using a multivariate Hidden Markov Model to infer latent market states and dynamically adjust exposure.

The strategy prioritizes:

- Stability over aggressiveness
- Risk-adjusted performance over absolute returns
- Robust validation over in-sample optimization

---

## Strategy Logic

### 1. Regime Detection

- Multivariate Gaussian HMM
- Trained using an **expanding walk-forward window**
- No parameter look-ahead or leakage
- Latent regimes inferred daily

### 2. Signal Construction

- For each regime, estimate expected Sharpe ratio
- Smooth expected Sharpe through time
- Map expected Sharpe to position size using a sigmoid function

### 3. Risk Filters

- **Trend filter**: EMA fast vs EMA slow
- **Volatility targeting**: normalize risk across regimes and time

### 4. Portfolio Output

- Continuous exposure between 0 and 1.2
- Daily strategy returns
- Equity curve and drawdown tracking

---

## Validation Methodology

This project explicitly avoids in-sample-only evaluation.

### Walk-Forward Backtest

- Model retrained yearly using only past data
- Strategy applied strictly out-of-sample
- Ensures realistic deployment conditions

### Block Bootstrap Monte Carlo

- Stationary block bootstrap to preserve serial dependence
- Thousands of resampled equity paths
- Empirical distributions of:
  - Sharpe ratio
  - CAGR
  - Volatility
  - Maximum drawdown

### Deflated Sharpe Ratio (DSR)

- Adjusts Sharpe significance for:
  - Non-normal returns (skewness, kurtosis)
  - Multiple strategy trials / data-snooping
- Provides a true statistical confidence level

---

## Results Summary

- Raw returns do not outperform Buy & Hold, as expected due to reduced market exposure.
- At matched volatility, the strategy achieves **comparable performance** to Buy & Hold.
- Maximum drawdowns are **significantly reduced**, particularly during crisis periods (2020, 2022).
- Block bootstrap Monte Carlo confirms robustness across resampled paths.
- Deflated Sharpe Ratio remains **statistically significant** after accounting for data-snooping.

The primary value of the strategy lies in **drawdown mitigation and risk control**, not in return amplification.

---

## Visual Analysis

The following figures are generated automatically:

- **Equity curve (raw)**  
  Absolute performance compared to Buy & Hold.

- **Equity curve (volatility-matched)**  
  Buy & Hold scaled to the same volatility as the strategy for fair comparison.

- **Drawdown comparison**  
  Downside risk, depth, and recovery speed across market regimes.

All figures are saved under:

reports/figures/

---

## Project Structure

src/
    hmm_strategy.py
    bootstrap.py
    deflated_sharpe.py
    export_figures.py

---

## How to Run

python src/hmm_strategy.py
python src/bootstrap.py
python src/deflated_sharpe.py
python src/export_figures.py

---

Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.


