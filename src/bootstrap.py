# bootstrap_to_excel.py
# Monte Carlo block bootstrap -> genera un Excel con resumen y stats
# Lee Excel/CSV de la estrategia y escribe un Excel de resultados automáticamente.

import os
import math
import numpy as np
import pandas as pd


# =========================
# CONFIG (EDITA ESTO)
# =========================
INPUT_PATH = r"hmm_QQQ_3estados_expSharpe_trend_voltarget_walkforward_multivar_BASE.xlsx"
SHEET_NAME = None          # None = primera hoja del Excel
RET_COL = "strat_returns"  # <-- cambia si tu columna tiene otro nombre

OUTPUT_PATH = "bootstrap_report.xlsx"  # Excel que creará el script

N_SIMS = 5000
BLOCK_LENS = [10, 20, 40]
SEED = 123
PERIODS_PER_YEAR = 252

# Opcional: guardar algunas curvas de equity (no todas, para que no pese una barbaridad)
SAVE_EQUITY_PATHS = True
N_EQUITY_PATHS_TO_SAVE = 50   # guarda 50 paths por block_len
EQUITY_PATHS_FREQ = "D"       # solo etiqueta, no afecta al cálculo


# =========================
# MÉTRICAS
# =========================
def equity_from_returns(r: np.ndarray) -> np.ndarray:
    # Retornos simples diarios
    return np.cumprod(1.0 + r)

def max_drawdown_from_equity(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))

def sharpe_ann(r: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(r)
    mu = np.mean(r)
    sd = np.std(r, ddof=1)
    if sd <= 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * math.sqrt(periods_per_year))

def vol_ann(r: np.ndarray, periods_per_year: int = 252) -> float:
    sd = np.std(r, ddof=1)
    if sd <= 0 or np.isnan(sd):
        return float("nan")
    return float(sd * math.sqrt(periods_per_year))

def cagr_from_equity(equity: np.ndarray, periods_per_year: int = 252) -> float:
    T = len(equity)
    years = T / periods_per_year
    if years <= 0:
        return float("nan")
    return float(equity[-1] ** (1.0 / years) - 1.0)


# =========================
# I/O
# =========================
def read_returns(path: str, ret_col: str, sheet_name=None) -> pd.Series:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext in [".csv"]:
        df = pd.read_csv(path)
    else:
        raise ValueError("Formato no soportado. Usa .xlsx/.xls o .csv")

    if ret_col not in df.columns:
        raise KeyError(
            f"No encuentro la columna '{ret_col}'. Columnas disponibles:\n{list(df.columns)}"
        )

    s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    s = s[np.isfinite(s)]

    if len(s) < 300:
        print(f"⚠️ Aviso: solo hay {len(s)} retornos válidos; conclusiones menos robustas.")
    return s


# =========================
# BOOTSTRAP
# =========================
def block_bootstrap_paths(returns: np.ndarray, n_sims: int, block_len: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns)
    T = len(returns)
    if T < block_len:
        raise ValueError(f"T={T} < block_len={block_len}. Reduce block_len.")

    n_blocks = int(np.ceil(T / block_len))
    sims = np.empty((n_sims, T), dtype=float)

    for i in range(n_sims):
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        path = np.concatenate([returns[s:s + block_len] for s in starts])[:T]
        sims[i] = path

    return sims

def summarize_sims(sims: np.ndarray, periods_per_year: int = 252) -> pd.DataFrame:
    n_sims, _ = sims.shape
    out = {
        "final_return": np.empty(n_sims),
        "cagr": np.empty(n_sims),
        "sharpe": np.empty(n_sims),
        "vol_ann": np.empty(n_sims),
        "max_dd": np.empty(n_sims),
    }

    for i in range(n_sims):
        r = sims[i]
        eq = equity_from_returns(r)
        out["final_return"][i] = eq[-1] - 1.0
        out["cagr"][i] = cagr_from_equity(eq, periods_per_year)
        out["sharpe"][i] = sharpe_ann(r, periods_per_year)
        out["vol_ann"][i] = vol_ann(r, periods_per_year)
        out["max_dd"][i] = max_drawdown_from_equity(eq)

    return pd.DataFrame(out)

def observed_metrics(r: np.ndarray, periods_per_year: int = 252) -> dict:
    eq = equity_from_returns(r)
    return {
        "final_return": float(eq[-1] - 1.0),
        "cagr": float(cagr_from_equity(eq, periods_per_year)),
        "sharpe": float(sharpe_ann(r, periods_per_year)),
        "vol_ann": float(vol_ann(r, periods_per_year)),
        "max_dd": float(max_drawdown_from_equity(eq)),
        "n_days": int(len(r)),
    }

def bootstrap_summary(stats: pd.DataFrame, obs: dict) -> dict:
    # percentiles
    pct = lambda col, p: float(np.nanpercentile(stats[col].values, p))

    # probabilidades
    p_sh_pos = float((stats["sharpe"] > 0).mean())
    p_fin_pos = float((stats["final_return"] > 0).mean())
    p_dd_20 = float((stats["max_dd"] < -0.20).mean())
    p_dd_30 = float((stats["max_dd"] < -0.30).mean())

    # percentil del Sharpe observado dentro del bootstrap
    sharpe_rank = float((stats["sharpe"] <= obs["sharpe"]).mean())

    return {
        "obs_final_return": obs["final_return"],
        "obs_cagr": obs["cagr"],
        "obs_sharpe": obs["sharpe"],
        "obs_vol_ann": obs["vol_ann"],
        "obs_max_dd": obs["max_dd"],
        "p5_final_return": pct("final_return", 5),
        "p50_final_return": pct("final_return", 50),
        "p95_final_return": pct("final_return", 95),
        "p5_cagr": pct("cagr", 5),
        "p50_cagr": pct("cagr", 50),
        "p95_cagr": pct("cagr", 95),
        "p5_sharpe": pct("sharpe", 5),
        "p50_sharpe": pct("sharpe", 50),
        "p95_sharpe": pct("sharpe", 95),
        "p5_vol_ann": pct("vol_ann", 5),
        "p50_vol_ann": pct("vol_ann", 50),
        "p95_vol_ann": pct("vol_ann", 95),
        "p5_max_dd": pct("max_dd", 5),
        "p50_max_dd": pct("max_dd", 50),
        "p95_max_dd": pct("max_dd", 95),
        "P_sharpe_gt_0": p_sh_pos,
        "P_final_gt_0": p_fin_pos,
        "P_maxdd_lt_-20": p_dd_20,
        "P_maxdd_lt_-30": p_dd_30,
        "obs_sharpe_percentile_in_bootstrap": sharpe_rank,
    }


# =========================
# MAIN -> ESCRIBE EXCEL
# =========================
def main():
    s = read_returns(INPUT_PATH, RET_COL, sheet_name=SHEET_NAME)
    r_obs = s.values.astype(float)

    obs = observed_metrics(r_obs, PERIODS_PER_YEAR)

    rows_summary = []
    stats_by_L = {}

    rng = np.random.default_rng(SEED)

    for L in BLOCK_LENS:
        sims = block_bootstrap_paths(r_obs, n_sims=N_SIMS, block_len=L, seed=SEED)
        stats = summarize_sims(sims, periods_per_year=PERIODS_PER_YEAR)
        stats_by_L[L] = (sims, stats)

        summ = bootstrap_summary(stats, obs)
        summ["block_len"] = L
        summ["n_sims"] = N_SIMS
        summ["n_days"] = obs["n_days"]
        rows_summary.append(summ)

    df_summary = pd.DataFrame(rows_summary).sort_values("block_len").reset_index(drop=True)

    # Escribir Excel
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        # Hoja resumen
        df_summary.to_excel(writer, sheet_name="SUMMARY", index=False)

        # Hoja observada (serie de retornos)
        pd.DataFrame({RET_COL: r_obs}).to_excel(writer, sheet_name="OBSERVED_RETURNS", index=False)

        # Por cada block_len: stats completos
        for L in BLOCK_LENS:
            sims, stats = stats_by_L[L]
            sheet_stats = f"STATS_L{L}"
            stats.to_excel(writer, sheet_name=sheet_stats, index=False)

            # Opcional: guardar algunas curvas de equity para inspección
            if SAVE_EQUITY_PATHS:
                k = min(N_EQUITY_PATHS_TO_SAVE, sims.shape[0])
                idx = rng.choice(sims.shape[0], size=k, replace=False)
                eqs = []
                for j in idx:
                    eq = equity_from_returns(sims[j])
                    eqs.append(eq)
                eqs = np.column_stack(eqs)  # (T, k)

                df_eq = pd.DataFrame(eqs, columns=[f"eq_sim_{i+1}" for i in range(k)])
                df_eq.insert(0, "t", np.arange(len(r_obs)))
                df_eq.to_excel(writer, sheet_name=f"EQUITY_L{L}", index=False)

    print(f"\n✅ Listo. Excel generado: {OUTPUT_PATH}")
    print("Abre la hoja SUMMARY y mira especialmente:")
    print("- P(Sharpe > 0)")
    print("- p5/p50/p95 de Sharpe y MaxDD")
    print("- obs_sharpe_percentile_in_bootstrap (si es >0.95, tu Sharpe histórico fue 'muy afortunado')")

if __name__ == "__main__":
    main()
