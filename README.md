\# HMM Regime + Trend + Vol Targeting Strategy



This repository implements a \*\*walk-forward Hidden Markov Model (HMM) regime strategy\*\*

combined with:

\- expected Sharpe–based positioning,

\- a medium-term trend filter,

\- volatility targeting,

\- and realistic execution constraints.



The strategy is evaluated across multiple ETF pairs using \*\*GLD as the risk asset\*\*

and a set of equity ETFs as the alternative leg.



---



\## Strategy Overview



At a high level, the strategy:



1\. \*\*Downloads daily prices\*\* using `yfinance`

2\. \*\*Builds HMM regimes\*\* on:

&nbsp;  - daily returns  

&nbsp;  - absolute returns (volatility proxy)

3\. \*\*Trains the HMM walk-forward\*\*, re-estimating parameters year by year  

&nbsp;  (no look-ahead bias in regime parameters)

4\. \*\*Forms expected Sharpe ratios\*\* using predictive regime probabilities

5\. \*\*Maps expected Sharpe to exposure\*\* via a sigmoid function

6\. \*\*Applies a trend filter\*\* (EMA fast vs slow)

7\. \*\*Targets constant volatility\*\*

8\. \*\*Executes with realistic frictions\*\*:

&nbsp;  - weekly rebalancing

&nbsp;  - dead-band (no-trade zone)

&nbsp;  - transaction costs proportional to turnover



---



\## Benchmarking Methodology



To avoid unfair comparisons, performance is measured against:

\- Buy \& Hold GLD

\- Buy \& Hold of the paired ETF

\- A \*\*static ex-ante benchmark mix\*\*:

&nbsp; - calibrated only on the \*\*training window\*\*

&nbsp; - matched to the \*\*strategy’s realized volatility in training\*\*

&nbsp; - fixed weights applied out-of-sample



This ensures that any alpha is not simply due to higher risk.



---



\## Empirical Results (2010 → Present)



\### Linear Alpha Regression (Daily, HAC / Newey-West)



Annualized alpha estimates were obtained via regression of strategy returns on:

\- GLD returns

\- Paired ETF returns



\#### Summary Across Pairs



| Pair (GLD vs) | Annualized Alpha | Alpha (with costs) | t-stat | t-stat (with costs) | p-value | p-value (with costs) |

|--------------|-----------------|--------------------|--------|---------------------|--------|----------------------|

| SPY | 1.48% | 0.59% | 1.23 | 0.50 | 0.218 | 0.620 |

| DIA | 1.55% | 0.66% | 1.32 | 0.56 | 0.186 | 0.573 |

| QQQ | 1.07% | 0.21% | 0.85 | 0.17 | 0.397 | 0.868 |

| IWM | 1.56% | 0.71% | 1.21 | 0.55 | 0.227 | 0.580 |

| EFA | 1.39% | 0.50% | 1.22 | 0.44 | 0.223 | 0.658 |

| EWJ | 1.08% | 0.19% | 0.89 | 0.16 | 0.375 | 0.874 |

| EWG | 1.87% | 1.01% | 1.42 | 0.77 | 0.154 | 0.440 |

| EWQ | 2.10% | 1.25% | 1.66 | 0.99 | 0.097 | 0.323 |

| EWI | 1.78% | 0.96% | 1.35 | 0.73 | 0.176 | 0.465 |

| EWP | 1.81% | 0.97% | 1.33 | 0.71 | 0.185 | 0.475 |

| FEZ | 2.04% | 1.19% | 1.59 | 0.93 | 0.111 | 0.351 |

| EWU | 1.23% | 0.35% | 1.09 | 0.31 | 0.274 | 0.755 |



---



\## Interpretation



\- \*\*Without transaction costs\*\*, the strategy:

&nbsp; - generates \*\*positive alpha across all pairs\*\*

&nbsp; - often outperforms the market on a risk-adjusted basis

\- \*\*With realistic transaction costs included\*\*:

&nbsp; - alpha is \*\*significantly reduced\*\*

&nbsp; - statistical significance largely disappears

\- This indicates that:

&nbsp; - the signal \*\*contains information\*\*

&nbsp; - but the edge is \*\*fragile to execution frictions\*\*



In other words, the model \*\*beats the market in principle\*\*, but \*\*turnover and costs are the main limiting factor\*\*.



---



\## Key Takeaways



\- Regime-based expected Sharpe signals are \*\*directionally correct\*\*

\- Trend + volatility targeting stabilizes risk effectively

\- Transaction costs dominate performance in daily/weekly implementations

\- Improving execution (lower turnover, slower signals, cheaper instruments)

&nbsp; is likely more impactful than improving the model itself



---



\## How to Run



\### Install dependencies

```bash

pip install -r requirements.txt



