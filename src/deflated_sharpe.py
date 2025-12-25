import math
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

INPUT_PATH = r"hmm_QQQ_3estados_expSharpe_trend_voltarget_walkforward_multivar_BASE.xlsx"
RET_COL = "strat_returns"
PERIODS_PER_YEAR = 252
N_TRIALS_LIST = [10, 25, 50]

def sharpe_ann(r, periods_per_year=252):
    mu = np.mean(r)
    sd = np.std(r, ddof=1)
    return (mu / sd) * math.sqrt(periods_per_year)

def read_returns(path, ret_col):
    sheets = pd.read_excel(path, sheet_name=None)
    df = next(iter(sheets.values()))
    return pd.to_numeric(df[ret_col], errors="coerce").dropna().values

def deflated_sharpe_ratio(returns, sharpe_obs, n_trials, periods_per_year=252):
    T = len(returns)
    s = skew(returns)
    k = kurtosis(returns, fisher=False)

    inside = 1 - s * sharpe_obs + ((k - 1) / 4.0) * sharpe_obs**2
    inside = max(1e-12, inside)
    sr_adj = sharpe_obs * math.sqrt(inside)

    z = norm.ppf(1 - 1 / n_trials)
    sr_crit = z * math.sqrt((periods_per_year - 1) / (T - 1))

    dsr = norm.cdf((sr_adj - sr_crit) * math.sqrt(T - 1))
    return dsr, sr_adj, sr_crit

def main():
    r = read_returns(INPUT_PATH, RET_COL)
    sr = sharpe_ann(r, PERIODS_PER_YEAR)

    print("=== Deflated Sharpe Ratio ===")
    print("T:", len(r))
    print("Sharpe:", round(sr, 4))
    print("Skew:", round(skew(r), 4))
    print("Kurtosis:", round(kurtosis(r, fisher=False), 4))

    for n_trials in N_TRIALS_LIST:
        dsr, sr_adj, sr_crit = deflated_sharpe_ratio(r, sr, n_trials, PERIODS_PER_YEAR)
        print(f"n_trials={n_trials}  SR_adj={sr_adj:.4f}  SR_crit={sr_crit:.4f}  DSR={dsr:.4f}")

if __name__ == "__main__":
    main()
