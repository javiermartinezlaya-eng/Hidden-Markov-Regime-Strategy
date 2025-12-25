import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === Ajusta solo si tus nombres de columnas son distintos ===
INPUT_XLSX = "hmm_QQQ_3estados_expSharpe_trend_voltarget_walkforward_multivar_BASE.xlsx"

# Estrategia
STRAT_RET_COL_CANDIDATES = ["strat_returns", "strategy_returns", "ret_strat", "returns_strat"]

# Benchmark (si no existe en el Excel, lo reconstruimos con yfinance si hay "close" o "adj_close";
# si tampoco, fallará con mensaje claro)
BH_RET_COL_CANDIDATES = ["returns", "ret_bh", "buyhold_returns", "qqq_returns", "returns_bh", "benchmark_returns"]

# Salida figuras
OUT_DIR = os.path.join("reports", "figures")


def _first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_first_sheet(path: str) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    df = next(iter(sheets.values()))
    # intenta usar columna fecha si existe
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    return df


def to_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_first_sheet(INPUT_XLSX)

    strat_col = _first_existing_col(df, STRAT_RET_COL_CANDIDATES)
    if strat_col is None:
        raise ValueError(
            f"No encuentro retornos de estrategia. Busqué {STRAT_RET_COL_CANDIDATES}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    bh_col = _first_existing_col(df, BH_RET_COL_CANDIDATES)

    r_strat = to_series(df, strat_col)

    if bh_col is None:
        # Intento reconstruir buy&hold si hay precio
        price_col = None
        for c in ["precio", "Adj Close", "adj_close", "Close", "close", "price", "Price"]:
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            raise ValueError(
                f"No encuentro retornos buy&hold ni precio para reconstruirlo.\n"
                f"Busqué returns {BH_RET_COL_CANDIDATES} y precios típicos.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
        px = pd.to_numeric(df[price_col], errors="coerce").dropna()
        r_bh = px.pct_change().dropna()
    else:
        r_bh = to_series(df, bh_col)

    # Alinear índices
    r = pd.concat([r_strat.rename("strat"), r_bh.rename("bh")], axis=1).dropna()
    r_strat = r["strat"]
    r_bh = r["bh"]

    # === 1) Equity normal ===
    eq_strat = equity_curve(r_strat)
    eq_bh = equity_curve(r_bh)

    plt.figure()
    plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold (QQQ)")
    plt.plot(eq_strat.index, eq_strat.values, label="Strategy (HMM)")
    plt.title("Equity curve (raw)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_raw.png"), dpi=160)
    plt.close()

    # === 2) Equity con volatilidad igualada ===
    vol_strat = annualized_vol(r_strat)
    vol_bh = annualized_vol(r_bh)
    scale = vol_strat / vol_bh if vol_bh > 0 else 1.0

    r_bh_scaled = r_bh * scale
    eq_bh_scaled = equity_curve(r_bh_scaled)

    plt.figure()
    plt.plot(eq_bh_scaled.index, eq_bh_scaled.values, label=f"Buy & Hold scaled to Strategy vol (x{scale:.3f})")
    plt.plot(eq_strat.index, eq_strat.values, label="Strategy (HMM)")
    plt.title("Equity curve (volatility-matched)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_vol_matched.png"), dpi=160)
    plt.close()

    # === 3) Drawdown ===
    dd_strat = drawdown_curve(eq_strat)
    dd_bh = drawdown_curve(eq_bh)

    plt.figure()
    plt.plot(dd_bh.index, dd_bh.values, label="Buy & Hold (QQQ)")
    plt.plot(dd_strat.index, dd_strat.values, label="Strategy (HMM)")
    plt.title("Drawdown comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "drawdown.png"), dpi=160)
    plt.close()

    print("✅ Figuras guardadas en:", OUT_DIR)
    print("- equity_raw.png")
    print("- equity_vol_matched.png")
    print("- drawdown.png")
    print(f"Info: vol_strat={vol_strat:.4f}, vol_bh={vol_bh:.4f}, scale={scale:.3f}")


if __name__ == "__main__":
    main()
