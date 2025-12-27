import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
# ============================================================
# 1. Parámetros
# ============================================================
TICKER = "QQQ"
START_DATE = "2010-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")
N_ESTADOS = 3
MAX_ITER = 50
FREQ_ANUAL = 252  
EMA_FAST = 20
EMA_SLOW = 120

# Sharpe esperado diario -> posición
EXP_SHARPE_SCALE = 0.2     
EXP_SHARPE_CUTOFF = 0.05   

# Volatility targeting
TARGET_VOL_ANUAL = 0.14     
ROLLING_VOL_DIAS = 20       
MAX_LEVERAGE = 1.35          

# Filtro de tendencia: peso mínimo/máximo
TREND_MIN_WEIGHT = 0.0      
TREND_MAX_WEIGHT = 1.0      

# ============================================================
# 2. Descargar precios
# ============================================================
def descargar_precios(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(data, pd.DataFrame):
        if "Close" in data.columns:
            close = data["Close"].dropna()
        else:
            close = data.iloc[:, 0].dropna()
    else:
        close = data.dropna()

    if isinstance(close, pd.Series):
        close = close.to_frame(name="Close")
    else:
        first_col = close.columns[0]
        if first_col != "Close":
            close = close.rename(columns={first_col: "Close"})
        close = close[["Close"]]

    return close

# ============================================================
# 3. HMM gaussiano multivariante con Baum–Welch (entrenamiento)
#    Observaciones: [r_t, |r_t|]
# ============================================================
def entrenar_hmm_probabilistico(observaciones, n_states=N_ESTADOS, max_iter=50,
                                random_state=1, tol=1e-6):
    """
    Entrena un HMM gaussiano multivariante (2D: r, |r|) usando Baum–Welch.

    observaciones: DataFrame/array de shape (T, 2)
    Devuelve:
      - posterior_filt_df  : P(s_t | obs_1..t)
      - posterior_smooth_df: P(s_t | obs_1..T)
      - medias_full        : array (n_states, 2) con medias por estado
      - covs               : array (n_states, 2, 2) con covarianzas por estado
      - A                  : matriz de transición
      - pi                 : distribución inicial
    """
    rng = np.random.default_rng(random_state)

    obs = np.asarray(observaciones.values, dtype=float)
    T = obs.shape[0]
    d = obs.shape[1]
    n_states = int(n_states)

    if T < n_states:
        raise ValueError("No hay suficientes observaciones para el número de estados elegido.")

    # Inicialización de medias
    indices = rng.choice(T, size=n_states, replace=False)
    medias_full = obs[indices, :].copy()

    # Covarianza global inicial
    if T > 1:
        global_cov = np.cov(obs.T)
    else:
        global_cov = np.eye(d)
    if np.ndim(global_cov) == 0:
        global_cov = np.eye(d) * float(global_cov)
    global_cov = np.asarray(global_cov, dtype=float)
    global_cov += np.eye(d) * 1e-6  # regularización

    covs = np.array([global_cov.copy() for _ in range(n_states)], dtype=float)

    # pi y A uniformes
    pi = np.full(n_states, 1.0 / n_states, dtype=float)
    A = np.full((n_states, n_states), 1.0 / n_states, dtype=float)

    loglik_prev = -np.inf

    for _ in range(max_iter):
        # ======================
        # E-step: B, alpha, beta, gamma, xi
        # ======================
        B = np.zeros((T, n_states), dtype=float)

        for k in range(n_states):
            mu_k = medias_full[k]
            Sigma_k = covs[k]

            Sigma_k = (Sigma_k + Sigma_k.T) / 2.0
            Sigma_k += np.eye(d) * 1e-8

            try:
                inv_Sigma_k = np.linalg.inv(Sigma_k)
                det_Sigma_k = np.linalg.det(Sigma_k)
            except np.linalg.LinAlgError:
                Sigma_k += np.eye(d) * 1e-6
                inv_Sigma_k = np.linalg.inv(Sigma_k)
                det_Sigma_k = np.linalg.det(Sigma_k)

            if det_Sigma_k <= 0:
                Sigma_k += np.eye(d) * 1e-6
                inv_Sigma_k = np.linalg.inv(Sigma_k)
                det_Sigma_k = np.linalg.det(Sigma_k)

            diff = obs - mu_k  # (T, d)
            mahal = np.einsum("ij,jk,ik->i", diff, inv_Sigma_k, diff)
            coeff = 1.0 / np.sqrt(((2.0 * np.pi) ** d) * det_Sigma_k)
            B[:, k] = coeff * np.exp(-0.5 * mahal)

        B = np.clip(B, 1e-300, None)

        alpha = np.zeros((T, n_states), dtype=float)
        c = np.zeros(T, dtype=float)

        alpha[0, :] = pi * B[0, :]
        c[0] = alpha[0, :].sum()
        if c[0] == 0.0:
            c[0] = 1e-300
        alpha[0, :] /= c[0]

        for t in range(1, T):
            alpha[t, :] = (alpha[t - 1, :] @ A) * B[t, :]
            c[t] = alpha[t, :].sum()
            if c[t] == 0.0:
                c[t] = 1e-300
            alpha[t, :] /= c[t]

        beta = np.zeros((T, n_states), dtype=float)
        beta[-1, :] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t, :] = A @ (B[t + 1, :] * beta[t + 1, :])
            beta[t, :] /= c[t + 1]

        gamma = alpha * beta
        gamma_sum_row = gamma.sum(axis=1, keepdims=True)
        gamma_sum_row[gamma_sum_row == 0.0] = 1e-300
        gamma /= gamma_sum_row

        xi = np.zeros((T - 1, n_states, n_states), dtype=float)
        for t in range(T - 1):
            numer = alpha[t, :, None] * A * (B[t + 1, :] * beta[t + 1, :])[None, :]
            denom = numer.sum()
            if denom == 0.0:
                denom = 1e-300
            xi[t, :, :] = numer / denom

        loglik = -np.sum(np.log(c))

        # ======================
        # M-step
        # ======================
        pi = gamma[0, :]

        sum_xi = xi.sum(axis=0)
        sum_gamma = gamma[:-1, :].sum(axis=0)

        for i in range(n_states):
            if sum_gamma[i] == 0.0:
                A[i, :] = 1.0 / n_states
            else:
                A[i, :] = sum_xi[i, :] / sum_gamma[i]

        A_row_sums = A.sum(axis=1, keepdims=True)
        A_row_sums[A_row_sums == 0.0] = 1.0
        A /= A_row_sums

        gamma_sum = gamma.sum(axis=0)

        for k in range(n_states):
            if gamma_sum[k] == 0.0:
                medias_full[k] = np.zeros(d)
                covs[k] = np.eye(d)
            else:
                w = gamma[:, k][:, None]
                mu_k = (w * obs).sum(axis=0) / gamma_sum[k]
                diff = obs - mu_k
                cov_k = (w * diff).T @ diff / gamma_sum[k]
                cov_k = (cov_k + cov_k.T) / 2.0
                cov_k += np.eye(d) * 1e-8
                medias_full[k] = mu_k
                covs[k] = cov_k

        if np.abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

    posterior_filt_df = pd.DataFrame(
        alpha,
        index=observaciones.index,
        columns=[f"state_{i}" for i in range(n_states)]
    )

    posterior_smooth_df = pd.DataFrame(
        gamma,
        index=observaciones.index,
        columns=[f"state_{i}" for i in range(n_states)]
    )

    return posterior_filt_df, posterior_smooth_df, medias_full, covs, A, pi

# ============================================================
# 3bis. Filtrado + suavizado con parámetros fijos (test)
# ============================================================
def filtrar_y_suavizar_hmm_dado_parametros(observaciones, medias_full, covs, A, pi):
    obs = np.asarray(observaciones.values, dtype=float)
    T = obs.shape[0]
    d = obs.shape[1]
    n_states = len(medias_full)

    B = np.zeros((T, n_states), dtype=float)

    for k in range(n_states):
        mu_k = medias_full[k]
        Sigma_k = covs[k]

        Sigma_k = (Sigma_k + Sigma_k.T) / 2.0
        Sigma_k += np.eye(d) * 1e-8

        inv_Sigma_k = np.linalg.inv(Sigma_k)
        det_Sigma_k = np.linalg.det(Sigma_k)
        if det_Sigma_k <= 0:
            Sigma_k += np.eye(d) * 1e-6
            inv_Sigma_k = np.linalg.inv(Sigma_k)
            det_Sigma_k = np.linalg.det(Sigma_k)

        diff = obs - mu_k
        mahal = np.einsum("ij,jk,ik->i", diff, inv_Sigma_k, diff)
        coeff = 1.0 / np.sqrt(((2.0 * np.pi) ** d) * det_Sigma_k)
        B[:, k] = coeff * np.exp(-0.5 * mahal)

    B = np.clip(B, 1e-300, None)

    alpha = np.zeros((T, n_states), dtype=float)
    c = np.zeros(T, dtype=float)

    alpha[0, :] = pi * B[0, :]
    c[0] = alpha[0, :].sum()
    if c[0] == 0.0:
        c[0] = 1e-300
    alpha[0, :] /= c[0]

    for t in range(1, T):
        alpha[t, :] = (alpha[t - 1, :] @ A) * B[t, :]
        c[t] = alpha[t, :].sum()
        if c[t] == 0.0:
            c[t] = 1e-300
        alpha[t, :] /= c[t]

    beta = np.zeros((T, n_states), dtype=float)
    beta[-1, :] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t, :] = A @ (B[t + 1, :] * beta[t + 1, :])
        beta[t, :] /= c[t + 1]

    gamma = alpha * beta
    gamma_sum_row = gamma.sum(axis=1, keepdims=True)
    gamma_sum_row[gamma_sum_row == 0.0] = 1e-300
    gamma /= gamma_sum_row

    posterior_filt_df = pd.DataFrame(
        alpha,
        index=observaciones.index,
        columns=[f"state_{i}" for i in range(n_states)]
    )
    posterior_smooth_df = pd.DataFrame(
        gamma,
        index=observaciones.index,
        columns=[f"state_{i}" for i in range(n_states)]
    )

    return posterior_filt_df, posterior_smooth_df

# ============================================================
# 3ter. Construir HMM walk-forward (sin look-ahead de parámetros)
# ============================================================
def construir_hmm_walkforward(observaciones, n_states=N_ESTADOS,
                              max_iter=MAX_ITER, random_state=42):
    """
    Devuelve:
    - posterior_filt_all  : P(s_t | obs_1..t) walk-forward
    - posterior_smooth_all: P(s_t | obs_1..T) (por año, sólo análisis)
    - medias_df_all       : medias del retorno por estado (por fecha)
    - sigmas_df_all       : sigmas del retorno por estado (por fecha)
    - pred_weights_all    : PREDICTIVO P(s_t | info hasta t-1) por fecha
    """
    years = sorted(observaciones.index.year.unique())
    if len(years) < 2:
        raise ValueError("Se necesita al menos 2 años de datos para hacer walk-forward.")

    posterior_filt_list = []
    posterior_smooth_list = []
    medias_df_list = []
    sigmas_df_list = []
    pred_weights_list = []

    for idx in range(1, len(years)):
        year = years[idx]
        mask_train = observaciones.index.year < year
        mask_test = observaciones.index.year == year

        obs_train = observaciones[mask_train]
        obs_test = observaciones[mask_test]

        if len(obs_train) < n_states + 10 or len(obs_test) == 0:
            continue

        print(f"\n[Walk-forward] Entrenando HMM multivariante con datos hasta año {year-1} "
              f"({obs_train.index.min().date()} -> {obs_train.index.max().date()}) "
              f"y aplicando a año {year}.")

        _, _, medias_full, covs, A, pi = entrenar_hmm_probabilistico(
            obs_train, n_states=n_states, max_iter=max_iter, random_state=random_state
        )

        posterior_filt_test, posterior_smooth_test = filtrar_y_suavizar_hmm_dado_parametros(
            obs_test, medias_full, covs, A, pi
        )

        posterior_filt_list.append(posterior_filt_test)
        posterior_smooth_list.append(posterior_smooth_test)

        # Medias y sigmas SOLO del retorno (dimensión 0)
        medias_r = medias_full[:, 0]
        sigmas_r = np.sqrt(covs[:, 0, 0])

        idx_test = obs_test.index
        cols = [f"state_{k}" for k in range(n_states)]

        medias_df_list.append(pd.DataFrame(np.tile(medias_r, (len(idx_test), 1)), index=idx_test, columns=cols))
        sigmas_df_list.append(pd.DataFrame(np.tile(sigmas_r, (len(idx_test), 1)), index=idx_test, columns=cols))

        # Pesos predictivos: P(s_t | info hasta t-1)
        alpha_vals = posterior_filt_test.values
        T_test, n_states_eff = alpha_vals.shape
        assert n_states_eff == n_states

        pred_weights = np.zeros_like(alpha_vals)
        pred_weights[0, :] = pi
        for t in range(1, T_test):
            pred_weights[t, :] = alpha_vals[t - 1, :] @ A

        pred_weights_list.append(pd.DataFrame(pred_weights, index=idx_test, columns=cols))

    if not posterior_filt_list:
        raise ValueError("No se pudo construir un posterior walk-forward (datos insuficientes).")

    posterior_filt_all = pd.concat(posterior_filt_list).sort_index()
    posterior_smooth_all = pd.concat(posterior_smooth_list).sort_index()
    medias_df_all = pd.concat(medias_df_list).sort_index()
    sigmas_df_all = pd.concat(sigmas_df_list).sort_index()
    pred_weights_all = pd.concat(pred_weights_list).sort_index()

    return posterior_filt_all, posterior_smooth_all, medias_df_all, sigmas_df_all, pred_weights_all

# ============================================================
# 4. Métricas
# ============================================================
def calcular_metricas(returns, freq_anual=252):
    equity = (1 + returns).cumprod()
    rent_acum = equity.iloc[-1] - 1

    vol_anual = returns.std() * np.sqrt(freq_anual)
    if vol_anual > 0 and np.isfinite(vol_anual):
        sharpe = returns.mean() * freq_anual / vol_anual
    else:
        sharpe = 0.0

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    metricas = {
        "rent_acum": rent_acum,
        "sharpe": sharpe,
        "vol_anual": vol_anual,
        "max_drawdown": max_dd
    }
    return metricas, equity, drawdown

# ============================================================
# 5. Estrategia HMM -> Sharpe esperado + tendencia + vol targeting
# ============================================================
def estrategia_hmm_prob_regime_trend(precios, posterior_filt, medias_df, sigmas_df,
                                     pred_weights, freq_anual=252):
    close = precios["Close"]
    returns = close.pct_change().fillna(0.0)

    posterior = posterior_filt.reindex(returns.index).ffill()
    medias_df = medias_df.reindex(returns.index).ffill()
    sigmas_df = sigmas_df.reindex(returns.index).ffill()
    pred_weights = pred_weights.reindex(returns.index).ffill()

    regimes_hard = posterior.values.argmax(axis=1)
    regimes_hard = pd.Series(regimes_hard, index=returns.index, name="regime_hard")

    medias_por_estado = returns.groupby(regimes_hard).mean().sort_index()
    print("\n=== Medias de retorno por estado HMM (duro, walk-forward multivariante) ===")
    for s in medias_por_estado.index:
        print(f"Estado {s}: media retorno diaria = {medias_por_estado[s]:.6f}")

    # Tendencia
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
    trend_flag = (ema_fast > ema_slow).astype(float)
    trend_flag_shifted = trend_flag.shift(1).fillna(0.0)

    # Sharpe esperado por mezcla HMM usando pesos predictivos
    medias_arr = medias_df.values
    sigmas_arr = sigmas_df.values
    weights_arr = pred_weights.values

    mu_mix = np.sum(weights_arr * medias_arr, axis=1)
    var_mix = np.sum(weights_arr * (sigmas_arr**2 + medias_arr**2), axis=1) - mu_mix**2
    var_mix = np.clip(var_mix, 1e-10, None)
    sigma_mix = np.sqrt(var_mix)

    exp_sharpe_daily = mu_mix / (sigma_mix + 1e-8)
    exp_sharpe_daily = pd.Series(exp_sharpe_daily, index=returns.index)

    exp_sharpe_smooth = exp_sharpe_daily.ewm(span=5, adjust=False).mean()
    exp_sharpe_shifted = exp_sharpe_smooth.shift(1)

    # Sigmoide
    k = 3.0
    x0 = EXP_SHARPE_CUTOFF
    scale = EXP_SHARPE_SCALE
    z = (exp_sharpe_shifted - x0) / scale
    base_position = 1.0 / (1.0 + np.exp(-k * z))
    base_position = pd.Series(base_position, index=returns.index)

    # Filtro tendencia
    trend_adjust = TREND_MIN_WEIGHT + (TREND_MAX_WEIGHT - TREND_MIN_WEIGHT) * trend_flag_shifted
    position_pre_vol = base_position * trend_adjust

    # Vol targeting
    rolling_vol = returns.rolling(ROLLING_VOL_DIAS).std() * np.sqrt(freq_anual)
    vol_scalar = TARGET_VOL_ANUAL / (rolling_vol + 1e-8)
    vol_scalar = vol_scalar.clip(0.0, MAX_LEVERAGE)
    vol_scalar = vol_scalar.shift(1).fillna(1.0)

    position = (position_pre_vol * vol_scalar).clip(0.0, MAX_LEVERAGE)

    strat_returns = position * returns
    metricas, equity, drawdown = calcular_metricas(strat_returns, freq_anual=freq_anual)

    df_resultados = pd.DataFrame({
        "precio": close,
        "returns": returns,
        "regime_hard": regimes_hard,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "trend_flag": trend_flag,
        "exp_sharpe_daily": exp_sharpe_daily,
        "exp_sharpe_shifted": exp_sharpe_shifted,
        "exp_sharpe_smooth": exp_sharpe_smooth,
        "base_position_sigmoid": base_position,
        "trend_adjust": trend_adjust,
        "position_pre_vol": position_pre_vol,
        "rolling_vol_anual": rolling_vol,
        "vol_scalar": vol_scalar,
        "position": position,
        "strat_returns": strat_returns,
        "equity": equity,
        "drawdown": drawdown
    })

    # Correlación y regresión simple (Pearson)
    serie_x = df_resultados["exp_sharpe_shifted"]
    serie_y = df_resultados["strat_returns"]
    mask = serie_x.notna() & serie_y.notna()

    if mask.sum() >= 2:
        x_vals = serie_x[mask].values
        y_vals = serie_y[mask].values
        corr_exp_sharp_shifted_returns = np.corrcoef(x_vals, y_vals)[0, 1]
        beta, alpha = np.polyfit(x_vals, y_vals, 1)
        y_hat = alpha + beta * x_vals
        ss_res = np.sum((y_vals - y_hat) ** 2)
        ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        corr_exp_sharp_shifted_returns = np.nan
        alpha = np.nan
        beta = np.nan
        r2 = np.nan

    return df_resultados, metricas, corr_exp_sharp_shifted_returns, alpha, beta, r2

# ============================================================
# 6. Análisis de lag (solo Pearson)
# ============================================================
def analizar_lag_exp_sharpe(df_resultados, max_lag=10):
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


    plt.figure(figsize=(10, 5))
    plt.plot(df_lags["lag"], df_lags["pearson"], marker="o", label="Pearson")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Lag (days, exp_sharpe_shifted offset)")
    plt.ylabel("Correlation (Pearson)")
    plt.title("Lag Sweep: exp_sharpe_shifted vs strat_returns (Pearson)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if df_lags["pearson"].abs().max() > 0:
        best_pear = df_lags.iloc[df_lags["pearson"].abs().idxmax()]
        print("\nMejor lag por |Pearson|:")
        print(best_pear)

    return df_lags

# ============================================================
# 7. Visualización de gráficas.
# ============================================================
def plot_sigmoid_and_nextday(df_resultados, max_scatter_points=6000, n_bins=12,
                             clip_q=(0.01, 0.99), add_counts=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Señal ex-ante y retorno del día siguiente
    x = df_resultados["exp_sharpe_shifted"].copy()
    y_next = df_resultados["returns"].shift(-1)              # retorno QQQ día siguiente
    base_pos = df_resultados["base_position_sigmoid"].copy()
    pos = df_resultados["position"].copy()

    d = pd.DataFrame({"x": x, "y_next": y_next, "base_pos": base_pos, "pos": pos}).dropna()
    if len(d) < 100:
        print("Muy pocos datos para graficar.")
        return
    # --------------------------------------------------------
    # 1) Scatter (downsample)
    # --------------------------------------------------------
    if len(d) > max_scatter_points:
        d_scatter = d.sample(max_scatter_points, random_state=1).sort_values("x")
    else:
        d_scatter = d.sort_values("x")

    plt.figure(figsize=(9, 5))
    plt.scatter(d_scatter["x"].values, d_scatter["y_next"].values, s=8, alpha=0.25)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(EXP_SHARPE_CUTOFF, linestyle="--", linewidth=1)
    plt.xlabel("exp_sharpe_shifted (ex-ante signal)")
    plt.ylabel("Next-day QQQ return (%) (returns[t+1])")
    plt.title("HMM expected Sharpe signal vs. next-day return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------------
    # 2) Bins CORREGIDOS: rangos fijos sobre señal winsorizada
    #    - clip_q limita colas para que no colapsen el eje X
    #    - bins uniformes en el rango central -> barras interpretables
    # --------------------------------------------------------
    ql, qh = clip_q
    x_lo = float(d["x"].quantile(ql))
    x_hi = float(d["x"].quantile(qh))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        # fallback seguro
        x_lo = float(d["x"].min())
        x_hi = float(d["x"].max())

    # Clip (winsorización) solo para construir bins y dibujar (no cambia y_next)
    d2 = d.copy()
    d2["x_clip"] = d2["x"].clip(x_lo, x_hi)

    # Bins uniformes
    edges = np.linspace(x_lo, x_hi, int(n_bins) + 1)
    d2["bin"] = pd.cut(d2["x_clip"], bins=edges, include_lowest=True)

    g = d2.groupby("bin", observed=True)

    # 2a) retorno medio del subyacente al día siguiente por bin
    mean_next = g["y_next"].mean()
    base_mid = g["base_pos"].median()
    counts = g.size()

    # geometría bins para pintar barras
    cats = mean_next.index.categories
    lefts = np.array([iv.left for iv in cats], dtype=float)
    rights = np.array([iv.right for iv in cats], dtype=float)
    centers = (lefts + rights) / 2.0
    widths = (rights - lefts) * 0.90

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(
        centers, mean_next.values,
        width=widths,
        align="center",
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2
    )
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.axvline(EXP_SHARPE_CUTOFF, linestyle="--", linewidth=1)
    ax1.set_xlabel(f"exp_sharpe_shifted (fixed bins, signal clipped to [{ql:.0%}, {qh:.0%}])")
    ax1.set_ylabel("Mean next-day QQQ return")
    ax1.set_title("Next-day return by signal bins + sigmoid (right axis)")
    ax1.grid(True, alpha=0.3)

    if add_counts:
        for cx, cy, n in zip(centers, mean_next.values, counts.values):
            ax1.text(cx, cy, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(centers, base_mid.values, marker="o", linestyle="--")
    ax2.set_ylabel("Base position (sigmoid)")

    plt.tight_layout()
    plt.show()
    # --------------------------------------------------------
    # 3) Bins: retorno "capturable" = position[t] * returns[t+1]
    # --------------------------------------------------------
    d2["capturable_next"] = d2["pos"] * d2["y_next"]
    mean_capturable = g["capturable_next"].mean()
    base_mid2 = g["base_pos"].median()
    counts2 = g.size()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(
        centers, mean_capturable.values,
        width=widths,
        align="center",
        color="tab:orange",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2
    )
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.axvline(EXP_SHARPE_CUTOFF, linestyle="--", linewidth=1)
    ax1.set_xlabel(f"exp_sharpe_shifted (fixed bins, signal clipped to [{ql:.0%}, {qh:.0%}])")
    ax1.set_ylabel("Average capturable return (position[t] × returns[t+1])")
    ax1.set_title("Next-day capturable return by signal bins (using position sizing)")
    ax1.grid(True, alpha=0.3)

    if add_counts:
        for cx, cy, n in zip(centers, mean_capturable.values, counts2.values):
            ax1.text(cx, cy, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(centers, base_mid2.values, marker="o", linestyle="--")
    ax2.set_ylabel("Base position (sigmoid)")
    plt.tight_layout()
    plt.show()
# ============================================================
# X. Histograma de retornos diarios + media de señal (HMM) y sigmoide por bin
# ============================================================
def plot_return_hist_with_signal_means(
    df_resultados,
    bin_width=0.02,          # 2% = 0.02
    r_min=-0.10,             # -10%
    r_max=0.10,              # +10%
    clip_outliers=True,
    show_table=True
):
    # Series base
    r = df_resultados["returns"].copy()
    x = df_resultados["exp_sharpe_shifted"].copy()
    b = df_resultados["base_position_sigmoid"].copy()

    d = pd.DataFrame({"ret": r, "exp": x, "base": b}).dropna()

    # Medias globales (LO NUEVO)
    mean_exp_global = d["exp"].mean()
    mean_base_global = d["base"].mean()

    # Recortar outliers
    if clip_outliers:
        d["ret"] = d["ret"].clip(lower=r_min, upper=r_max)

    # Crear bins fijos
    edges = np.arange(r_min, r_max + bin_width, bin_width)
    d["bin"] = pd.cut(d["ret"], bins=edges, include_lowest=True, right=False)

    # Estadísticos por bin
    stats = (
        d.groupby("bin")
         .agg(
             n=("ret", "size"),
             mean_ret=("ret", "mean"),
             mean_exp_sharpe=("exp", "mean"),
             mean_base_pos=("base", "mean")
         )
         .reset_index()
    )

    stats = stats[stats["n"] > 0].copy()
    # Centros de bin
    bin_lefts = np.array([iv.left for iv in stats["bin"]])
    bin_centers = bin_lefts + bin_width / 2.0
    # =======================
    # Gráfico
    # =======================
    fig, ax1 = plt.subplots(figsize=(11, 5))
    # Histograma (recuento)
    ax1.bar(
        bin_centers,
        stats["n"].values,
        width=bin_width * 0.9,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8
    )
    ax1.set_xlabel("Retorno diario QQQ (bins fijos)")
    ax1.set_ylabel("Número de días (n)")
    ax1.set_title(
        f"Daily returns histogram (bin = {bin_width*100:.0f}%) with mean signal per bin")
    ax1.grid(True, alpha=0.25)
    # Eje secundario (señales)
    ax2 = ax1.twinx()
    # Líneas por bin
    ax2.plot(
        bin_centers,
        stats["mean_exp_sharpe"].values,
        marker="o",
        linestyle="--",
        linewidth=1.2,
        label="Mean exp_sharpe_shifted (by bin)")
    ax2.plot(
        bin_centers,
        stats["mean_base_pos"].values,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        label="Average base_position_sigmoid per bin")
    # -----------------------
    # Líneas horizontales (medias globales)
    # -----------------------
    ax2.axhline(
        mean_exp_global,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=f"Global mean exp_sharpe ({mean_exp_global:.3f})")
    ax2.axhline(
        mean_base_global,
        color="tab:red",
        linestyle=":",
        linewidth=2.0,
        label=f"Global mean base_position ({mean_base_global:.3f})")

    ax2.set_ylabel("Mean signal / position")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    if show_table:
        out = stats.copy()
        out["bin"] = out["bin"].astype(str)
        print("\n=== Daily return statistics by bin ===")
        print(out.to_string(index=False))

    return stats

# ============================================================
# 8. Main
# ============================================================
def main():
    print(f"Descargando {TICKER}...")
    prices = descargar_precios(TICKER, START_DATE, END_DATE)

    r = prices["Close"].pct_change().dropna()
    obs = pd.DataFrame({"r": r, "v": r.abs()}, index=r.index)

    print("\nConstruyendo HMM walk-forward multivariante (sin look-ahead de parámetros)…")
    posterior_filt, posterior_smooth, medias_df, sigmas_df, pred_weights = construir_hmm_walkforward(
        obs, n_states=N_ESTADOS, max_iter=MAX_ITER, random_state=42
    )

    first_idx = posterior_filt.index.min()
    prices_sub = prices.loc[first_idx:]

    print("\nAplicando estrategia HMM_expSharpe + trend + vol targeting (walk-forward multivariante)…")
    df_resultados, metricas, corr_exp_sharp_shifted_returns, alpha, beta, r2 = estrategia_hmm_prob_regime_trend(
        prices_sub, posterior_filt, medias_df, sigmas_df, pred_weights, freq_anual=FREQ_ANUAL
    )

    print(f"\n=== Métricas Estrategia base HMM_expSharpe+Trend+VolTarget sobre {TICKER} ===")
    for k, v in metricas.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    print("\n=== Correlación (Pearson) entre exp_sharpe_shifted y strat_returns ===")
    try:
        print(f"Coeficiente de correlación de Pearson: {float(corr_exp_sharp_shifted_returns):.4f}")
    except Exception:
        print(f"Coeficiente de correlación de Pearson: {corr_exp_sharp_shifted_returns}")

    print("\n=== Regresión lineal: strat_returns_t = alpha + beta * exp_sharpe_shifted_t ===")
    print(f"alpha (intercepto): {alpha:.8f}")
    print(f"beta  (pendiente) : {beta:.8f}")
    print(f"R^2               : {r2:.4f}")
    
    # Lag Pearson
    df_lags = analizar_lag_exp_sharpe(df_resultados, max_lag=20)

    # Rentabilidad anual
    annual_base = df_resultados["strat_returns"].groupby(df_resultados.index.year).apply(lambda x: (1 + x).prod() - 1)
    df_annual = pd.DataFrame({"annual_base": annual_base})

    print("\n=== Rentabilidad anual (base) ===")
    print(df_annual)

    # Equity comparativa
    bh_returns = df_resultados["precio"].pct_change().fillna(0)
    bh_equity = (1 + bh_returns).cumprod()
    eq_curve = pd.DataFrame({
        "precio": df_resultados["precio"],
        "equity_buyhold": bh_equity,
        "equity_base": df_resultados["equity"],
    })
    # Heatmap mensual
    df_heat = df_resultados.copy()
    df_heat["year"] = df_heat.index.year
    df_heat["month"] = df_heat.index.month
    heatmap_base = df_heat.groupby(["year", "month"])["strat_returns"].apply(lambda x: (1 + x).prod() - 1).unstack()

    # Exportar a Excel
    file_name = f"hmm_{TICKER}_{N_ESTADOS}estados_expSharpe_trend_voltarget_walkforward_multivar_BASE.xlsx"
    with pd.ExcelWriter(file_name) as writer:
        df_resultados.to_excel(writer, sheet_name="Resultados_diarios_base")
        df_annual.to_excel(writer, sheet_name="Rentabilidad_anual_base")
        eq_curve.to_excel(writer, sheet_name="Equity_curve")
        heatmap_base.to_excel(writer, sheet_name="Heatmap_mensual_base")
        posterior_smooth.to_excel(writer, sheet_name="Posterior_suavizado_walkforward")
        pred_weights.to_excel(writer, sheet_name="Pred_weights_walkforward")
        df_lags.to_excel(writer, sheet_name="Lag_exp_sharpe_vs_strat")

    print(f"\nExcel guardado como: {file_name}")

    # Señal sugerida (última fecha)
    last_dt = df_resultados.index[-1]
    row = df_resultados.iloc[-1]
    print("\n=== Señal sugerida por el modelo (última fecha disponible) ===")
    print(f"Fecha señal: {last_dt.date()}")
    print("\n--- Filtro 1: HMM + sigmoide ---")
    print(f"Sharpe esperado suavizado (exp_sharpe_shifted): {row['exp_sharpe_shifted']:.4f}")
    print(f"Base position (sigmoide): {row['base_position_sigmoid']:.4f}")
    print("\n--- Filtro 2: Tendencia ---")
    print(f"EMA fast > EMA slow (trend_flag): {int(row['trend_flag'])}")
    print(f"Trend adjust aplicado (usa shift): {row['trend_adjust']:.2f}")
    print("\n--- Filtro 3: Volatility targeting ---")
    print(f"Vol anual estimada (rolling): {row['rolling_vol_anual']:.4f}")
    print(f"Vol scalar aplicado: {row['vol_scalar']:.4f}")
    print("\n=== RESULTADO FINAL ===")
    print(f"Posición sugerida por el modelo: {row['position']:.4f}")

    # Visualizaciones (sigmoide vs retorno siguiente / capturable)
    plot_sigmoid_and_nextday(df_resultados, max_scatter_points=6000, n_bins=12)

    # =========================================================
    # Histograma retornos diarios + medias de señal/sigmoide por bin de retorno
    # =========================================================
    plot_return_hist_with_signal_means(
        df_resultados,
        bin_width=0.0025,     # 1% en 1%
        r_min=-0.10,        # desde -10%
        r_max=0.10,         # hasta +10%
        clip_outliers=True,
        show_table=True
    )

if __name__ == "__main__":
    main()
