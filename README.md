Hidden Markov Regime Strategy (GLD Regime Signal)
================================================

Summary
-------

This repository implements a regime-aware allocation strategy where the “risk” asset is GLD and the second leg is a paired ETF (SPY, DIA, QQQ, IWM, EFA, EWJ, EWG, EWQ, EWI, EWP, FEZ, EWU). A multivariate Gaussian Hidden Markov Model (HMM) is trained in a strict walk-forward manner to infer latent regimes from GLD’s return dynamics, then converted into a long-only exposure to GLD with a complementary allocation to the paired ETF.

The implementation is intentionally constrained: no parameter optimization on test data, executable weights are lagged (no look-ahead), and transaction costs are explicitly modeled.

What This Code Does (Exact Behavior)
------------------------------------

Assets and Data
- Risk asset: GLD (RISK_TICKER = GLD)
- Paired assets: SPY, DIA, QQQ, IWM, EFA, EWJ, EWG, EWQ, EWI, EWP, FEZ, EWU
- Data source: yfinance (auto-adjusted prices)
- Sample start: 2010-01-01
- Sample end: today (runtime)

Model (HMM)
- Model: Multivariate Gaussian HMM trained via Baum–Welch
- Observations: two-dimensional features derived from GLD returns
  - r_t   = daily return of GLD
  - v_t   = |r_t| (absolute return as a volatility proxy)
- States: 3 (N_ESTADOS = 3)
- Walk-forward protocol (no look-ahead in parameters):
  - For each calendar year Y in the sample:
    - Train HMM on all data strictly before Y
    - Apply fixed parameters to filter/smooth probabilities for year Y
  - Concatenate yearly out-of-sample posteriors into a full walk-forward posterior

Signal Construction and Portfolio
---------------------------------

Expected Sharpe from predictive regime weights
- The strategy uses one-step-ahead predictive regime probabilities:
  - Predicted weights at time t are formed from the filtered posterior at t-1 and the transition matrix A
- Using predicted regime weights, the strategy computes a mixture mean and variance for GLD returns
- From that mixture, an expected daily Sharpe estimate is computed and smoothed

Position mapping and overlays
- Expected Sharpe is mapped to a long-only GLD weight using a sigmoid function:
  - EXP_SHARPE_CUTOFF and EXP_SHARPE_SCALE govern the threshold and sensitivity
- Trend filter: EMA(20) vs EMA(120) on GLD
  - Trend enters as a multiplicative weight (TREND_MIN_WEIGHT to TREND_MAX_WEIGHT)
- Volatility targeting:
  - Target annual volatility: 14% (TARGET_VOL_ANUAL = 0.14)
  - Rolling window: 20 days
  - Max leverage: 1.0 (long-only, capped)
- Turnover reduction:
  - Rebalance every 5 trading days (REBALANCE_EVERY = 5)
  - Dead-band: 5% absolute change in weight ignored (BAND = 0.05)
- Execution without look-ahead:
  - The weight applied to returns at time t is the weight decided at t-1

Transaction Costs (Explicitly Modeled)
--------------------------------------

- Costs are proportional to turnover:
  - Turnover is defined for a two-asset mix as 2 * |Δ w_GLD|
- Transaction cost rate: 15 bps per unit turnover (TC_BPS = 15)
- The code computes:
  - strat_returns_gross: portfolio return before costs
  - strat_returns: portfolio return after costs (net)

Benchmarks (Ex-Ante Fair Comparison)
------------------------------------

In addition to Buy-and-Hold GLD and Buy-and-Hold paired ETF, the code builds a static “BH mix ex-ante” benchmark:

- Calibration period: first 3 years (BENCH_TRAIN_YEARS = 3)
- Target volatility: the strategy’s realized volatility during the training period (ex-ante)
- Using only training data, the benchmark finds fixed weights (w_GLD, w_other) that match this target volatility
- The resulting fixed-weight mix is applied to the full sample

This benchmark is designed to compare the strategy against a static allocation at comparable risk, without using future information.

Alpha Estimation (As Used For the Results Table)
------------------------------------------------

For each GLD–ETF pair, alpha is estimated by linear regression with Newey–West (HAC) standard errors:

- Regression:
  - strategy_returns = const + b1 * GLD_returns + b2 * other_returns + error
- HAC configuration:
  - maxlags = 5
- Annualization:
  - annualized alpha = daily alpha * 252

The results below report alpha, t-statistics, and p-values both before and after transaction costs.

Empirical Results (GLD vs Paired ETF)
-------------------------------------

Pair (GLD & ETF) | Annualized Alpha | Annualized Alpha (With Costs) | t-stat | t-stat (With Costs) | p-value | p-value (With Costs)
--- | --- | --- | --- | --- | --- | ---
SPY | 1.48% | 0.59% | 1.23 | 0.50 | 0.218 | 0.620
DIA | 1.55% | 0.66% | 1.32 | 0.56 | 0.186 | 0.573
QQQ | 1.07% | 0.21% | 0.85 | 0.17 | 0.397 | 0.868
IWM | 1.56% | 0.71% | 1.21 | 0.55 | 0.227 | 0.580
EFA | 1.39% | 0.50% | 1.22 | 0.44 | 0.223 | 0.658
EWJ | 1.08% | 0.19% | 0.89 | 0.16 | 0.375 | 0.874
EWG | 1.87% | 1.01% | 1.42 | 0.77 | 0.154 | 0.440
EWQ | 2.10% | 1.25% | 1.66 | 0.99 | 0.097 | 0.323
EWI | 1.78% | 0.96% | 1.35 | 0.73 | 0.176 | 0.465
EWP | 1.81% | 0.97% | 1.33 | 0.71 | 0.185 | 0.475
FEZ | 2.04% | 1.19% | 1.59 | 0.93 | 0.111 | 0.351
EWU | 1.23% | 0.35% | 1.09 | 0.31 | 0.274 | 0.755

Notes on interpretation
- Alphas are positive across all pairs pre-costs, and materially reduced after costs.
- Statistical significance is not strong under conventional thresholds across the set.
- The strategy should be evaluated primarily as an exposure-control / risk-management overlay rather than a guaranteed alpha generator.

Figures and Outputs
-------------------

Strategy figures and diagnostics are saved under:

- reports/figures

These include equity curves comparing:
- Strategy (net)
- Buy-and-Hold GLD
- Buy-and-Hold paired ETF
- Ex-ante volatility-matched static mix benchmark

Installation
------------

Clone and install dependencies:

    git clone https://github.com/javiermartinezlaya-eng/Hidden-Markov-Regime-Strategy.git
    cd Hidden-Markov-Regime-Strategy
    pip install -r requirements.txt

Usage
-----

Run the full multi-pair backtest (GLD vs all tickers in PAIRS):

    python <your_script_filename>.py

The script will:
- download data,
- train the HMM in a walk-forward yearly scheme,
- compute strategy returns (gross and net),
- compute HAC regression alpha per pair,
- and display the equity curve plot for each pair.

Limitations
-----------

- Transaction costs are simplified (fixed bps per turnover) and do not include slippage/market impact.
- Daily close-to-close returns are assumed.
- Results depend on the chosen HMM state count, feature specification, and walk-forward granularity (yearly refits).
- The paired ETF is treated as the “safe” leg in the two-asset mix, which is a modeling choice and not cash.

Disclaimer
----------

This repository is provided solely for research and educational purposes.
Nothing contained herein constitutes investment advice, financial advice, trading advice, or a recommendation to buy or sell any securities, futures, or other financial instruments.

The author assumes no responsibility for any losses or damages resulting from the use of this code or any decisions made based on its outputs. Use at your own risk.

Past performance is not indicative of future results.

License
-------

MIT License. See LICENSE for details.



