Hidden Markov Regime Strategy
============================

Abstract
--------

This repository implements a quantitative trading research framework based on Hidden Markov Models (HMMs) for latent market regime identification. The objective is not to maximize absolute returns, but to evaluate whether regime-aware exposure control can improve drawdown behavior and risk-adjusted performance under strict out-of-sample and statistical validation.

The project prioritizes methodological rigor, robustness, and defensible inference over backtest optimization.


Overview
--------

Financial markets exhibit time-varying statistical properties that can be described as latent regimes with distinct volatility, trend persistence, and return characteristics. This project develops a systematic approach to:

- Infer latent market regimes using probabilistic state-space models
- Translate regime probabilities into trading exposure signals
- Enforce walk-forward evaluation to prevent look-ahead bias
- Quantify statistical significance while controlling for data-snooping effects


Methodology
-----------

Regime Inference
- Multivariate Gaussian Hidden Markov Model
- Rolling (walk-forward) training to ensure temporal integrity
- Daily regime probability estimation
- Regime count fixed ex-ante

Signal Construction
- Exposure derived from regime posterior probabilities
- Trend and volatility filters applied
- Volatility normalization to enable fair comparison with benchmarks

Validation and Statistical Inference
- Block bootstrap Monte Carlo simulations to preserve temporal dependence
- Empirical distributions for Sharpe ratio, CAGR, and drawdowns
- Deflated Sharpe Ratio applied to adjust for multiple testing and selection bias


Project Structure
-----------------

    ├── reports/
    │   └── figures/               Strategy outputs and diagnostics
    ├── src/
    │   ├── hmm_strategy.py        Core regime model and signal logic
    │   ├── bootstrap.py           Block bootstrap inference
    │   ├── deflated_sharpe.py     Statistical adjustment utilities
    │   └── export_figures.py      Figure generation
    ├── requirements.txt
    ├── README.md
    └── LICENSE


Installation
------------

Clone the repository and install dependencies:

    git clone https://github.com/javiermartinezlaya-eng/Hidden-Markov-Regime-Strategy.git
    cd Hidden-Markov-Regime-Strategy
    pip install -r requirements.txt

Use of a virtual environment or conda environment is recommended for reproducibility.


Usage
-----

Run the regime-based strategy:

    python src/hmm_strategy.py

Run bootstrap inference:

    python src/bootstrap.py

Generate figures and reports:

    python src/export_figures.py

Script parameters can be modified to adjust asset universe, rolling window length, feature selection, and regime count.


Results Interpretation
----------------------

- Raw returns are intentionally constrained due to volatility targeting
- Volatility-matched performance is comparable to, and in some periods exceeds, Buy-and-Hold
- Maximum drawdowns are materially reduced during high-volatility regimes
- Statistical conclusions are based on distributional inference rather than point estimates

Results should be interpreted in the context of risk control and robustness, not return maximization.


Limitations
-----------

- Transaction costs, slippage, and market impact are not modeled
- Execution assumes daily close-to-close trading
- Regime classification may become unstable during structural market breaks
- Performance is sensitive to regime count and feature specification


Reproducibility
---------------

- Walk-forward methodology enforces strict temporal ordering
- Bootstrap inference preserves serial correlation
- No parameter optimization is performed on test data
- Results are reproducible given identical data inputs and random seeds


Disclaimer
----------

This repository is provided solely for research and educational purposes.
Nothing contained herein constitutes investment advice, financial advice, trading advice, or a recommendation to buy or sell any securities, futures, or other financial instruments.

The author assumes no responsibility for any losses or damages resulting from the use of this code or any decisions made based on its outputs. Use at your own risk.

Past performance is not indicative of future results.


License
-------

This project is licensed under the MIT License. See the LICENSE file for details.



