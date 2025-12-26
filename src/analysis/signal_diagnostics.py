# src/analysis/signal_diagnostics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_or_show(fig, save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def analizar_lag_exp_sharpe(df_resultados, max_lag=20, save_path=None):
    exp_sharpe = df_resultados["exp_sharpe_shifted"]
    strat_returns = df_resultados["strat_returns"]

    lags = range(-max_lag, max_lag + 1)
    rows = []

    for lag in lags:
        shifted = exp_sharpe.shift(lag)
        mask = shifted.notna() & strat_returns.notna()
        if mask.sum() < 10:
            pear = np.nan
        else:
            x = shifted[mask].values
            y = strat_returns[mask].values
            pear = np.corrcoef(x, y)[0, 1]
        rows.append((lag, pear))

    df_lags = pd.DataFrame(rows, columns=["lag", "pearson"])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(df_lags["lag"], df_lags["pearson"], marker="o", label="Pearson")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Lag (días, exp_sharpe_shifted desplazada)")
    plt.ylabel("Correlación (Pearson)")
    plt.title("Lag sweep: exp_sharpe_shifted vs strat_returns (Pearson)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save_or_show(fig, save_path)

    return df_lags


def plot_sigmoid_and_nextday(
    df_resultados,
    max_scatter_points=6000,
    n_bins=12,
    clip_q=(0.01, 0.99),
    add_counts=True,
    cutoff=0.05,
    save_path_prefix=None,
):
    # Señal ex-ante y retorno del día siguiente
    x = df_resultados["exp_sharpe_shifted"].copy()
    y_next = df_resultados["returns"].shift(-1)  # retorno subyacente día siguiente
    base_pos = df_resultados["base_position_sigmoid"].copy()
    pos = df_resultados["position"].copy()

    d = pd.DataFrame({"x": x, "y_next": y_next, "base_pos": base_pos, "pos": pos}).dropna()
    if len(d) < 100:
        return

    # 1) Scatter
    if len(d) > max_scatter_points:
        d_scatter = d.sample(max_scatter_points, random_state=1).sort_values("x")
    else:
        d_scatter = d.sort_values("x")

    fig = plt.figure(figsize=(9, 5))
    plt.scatter(d_scatter["x"].values, d_scatter["y_next"].values, s=8, alpha=0.25)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(cutoff, linestyle="--", linewidth=1)
    plt.xlabel("exp_sharpe_shifted (señal ex-ante)")
    plt.ylabel("Retorno día siguiente (returns[t+1])")
    plt.title("Señal HMM (exp_sharpe) vs retorno del día siguiente")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    sp = None
    if save_path_prefix:
        sp = f"{save_path_prefix}_scatter.png"
    _save_or_show(fig, sp)

    # 2) Bins fijos sobre señal winsorizada
    ql, qh = clip_q
    x_lo = float(d["x"].quantile(ql))
    x_hi = float(d["x"].quantile(qh))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo = float(d["x"].min())
        x_hi = float(d["x"].max())

    d2 = d.copy()
    d2["x_clip"] = d2["x"].clip(x_lo, x_hi)

    edges = np.linspace(x_lo, x_hi, int(n_bins) + 1)
    d2["bin"] = pd.cut(d2["x_clip"], bins=edges, include_lowest=True)

    g = d2.groupby("bin", observed=True)
    mean_next = g["y_next"].mean()
    base_mid = g["base_pos"].median()
    counts = g.size()

    cats = mean_next.index.categories
    lefts = np.array([iv.left for iv in cats], dtype=float)
    rights = np.array([iv.right for iv in cats], dtype=float)
    centers = (lefts + rights) / 2.0
    widths = (rights - lefts) * 0.90

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(centers, mean_next.values, width=widths, align="center", alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.axvline(cutoff, linestyle="--", linewidth=1)
    ax1.set_xlabel(f"exp_sharpe_shifted (bins fijos, clip [{ql:.0%},{qh:.0%}])")
    ax1.set_ylabel("Retorno medio día siguiente")
    ax1.set_title("Retorno siguiente por bins de señal + sigmoide (eje derecho)")
    ax1.grid(True, alpha=0.3)

    if add_counts:
        for cx, cy, n in zip(centers, mean_next.values, counts.values):
            ax1.text(cx, cy, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(centers, base_mid.values, marker="o", linestyle="--")
    ax2.set_ylabel("Base position (sigmoide)")
    plt.tight_layout()

    sp = None
    if save_path_prefix:
        sp = f"{save_path_prefix}_bins_nextday.png"
    _save_or_show(fig, sp)

    # 3) Capturable = position[t] * returns[t+1]
    d2["capturable_next"] = d2["pos"] * d2["y_next"]
    mean_capturable = g["capturable_next"].mean()
    base_mid2 = g["base_pos"].median()
    counts2 = g.size()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(centers, mean_capturable.values, width=widths, align="center", alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.axvline(cutoff, linestyle="--", linewidth=1)
    ax1.set_xlabel(f"exp_sharpe_shifted (bins fijos, clip [{ql:.0%},{qh:.0%}])")
    ax1.set_ylabel("Retorno medio capturable (position[t] * returns[t+1])")
    ax1.set_title("Retorno capturable día siguiente por bins de señal")
    ax1.grid(True, alpha=0.3)

    if add_counts:
        for cx, cy, n in zip(centers, mean_capturable.values, counts2.values):
            ax1.text(cx, cy, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(centers, base_mid2.values, marker="o", linestyle="--")
    ax2.set_ylabel("Base position (sigmoide)")
    plt.tight_layout()

    sp = None
    if save_path_prefix:
        sp = f"{save_path_prefix}_bins_capturable.png"
    _save_or_show(fig, sp)


def plot_return_hist_with_signal_means(
    df_resultados,
    bin_width=0.0025,
    r_min=-0.10,
    r_max=0.10,
    clip_outliers=True,
    show_table=False,
    save_path=None,
):
    r = df_resultados["returns"].copy()
    x = df_resultados["exp_sharpe_shifted"].copy()
    b = df_resultados["base_position_sigmoid"].copy()

    d = pd.DataFrame({"ret": r, "exp": x, "base": b}).dropna()

    mean_exp_global = d["exp"].mean()
    mean_base_global = d["base"].mean()

    if clip_outliers:
        d["ret"] = d["ret"].clip(lower=r_min, upper=r_max)

    edges = np.arange(r_min, r_max + bin_width, bin_width)
    d["bin"] = pd.cut(d["ret"], bins=edges, include_lowest=True, right=False)

    stats = (
        d.groupby("bin", observed=True)
         .agg(
             n=("ret", "size"),
             mean_ret=("ret", "mean"),
             mean_exp_sharpe=("exp", "mean"),
             mean_base_pos=("base", "mean")
         )
         .reset_index()
    )
    stats = stats[stats["n"] > 0].copy()

    bin_lefts = np.array([iv.left for iv in stats["bin"]])
    bin_centers = bin_lefts + bin_width / 2.0

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(bin_centers, stats["n"].values, width=bin_width * 0.9, edgecolor="black", linewidth=0.8, alpha=0.8)
    ax1.set_xlabel("Retorno diario (bins fijos)")
    ax1.set_ylabel("Número de días (n)")
    ax1.set_title(f"Histograma retornos (bin={bin_width*100:.2f}%) + señal media por bin")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(bin_centers, stats["mean_exp_sharpe"].values, marker="o", linestyle="--", linewidth=1.2, label="Media exp_sharpe (por bin)")
    ax2.plot(bin_centers, stats["mean_base_pos"].values, marker="o", linestyle="-", linewidth=1.2, label="Media base_position (por bin)")

    ax2.axhline(mean_exp_global, linestyle=":", linewidth=2.0, label=f"Media global exp_sharpe ({mean_exp_global:.3f})")
    ax2.axhline(mean_base_global, linestyle=":", linewidth=2.0, label=f"Media global base_position ({mean_base_global:.3f})")

    ax2.set_ylabel("Media señal / posición")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    _save_or_show(fig, save_path)

    if show_table:
        out = stats.copy()
        out["bin"] = out["bin"].astype(str)
        print(out.to_string(index=False))

    return stats
