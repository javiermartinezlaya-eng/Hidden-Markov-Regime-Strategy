import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm

HMM_TICKER = "GLD"  
PAIRS = [ "SPY",   "DIA",   "QQQ",  "IWM",   "EFA", "EWJ",   "EWG",  "EWQ", "EWI",   "EWP",  "FEZ",  "EWU", 
        ]

START_DATE = "2010-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# ============================================================
# 1) Parámetros del modelo
# ============================================================
N_ESTADOS = 3
MAX_ITER = 50
FREQ_ANUAL = 252

EMA_FAST = 20
EMA_SLOW = 120

EXP_SHARPE_SCALE = 0.2
EXP_SHARPE_CUTOFF = 0.05

TARGET_VOL_ANUAL = 0.14
ROLLING_VOL_DIAS = 20
MAX_LEVERAGE = 1

TREND_MIN_WEIGHT = 0.0
TREND_MAX_WEIGHT = 1.0

# ============================================================
# BENCHMARK 
# ============================================================
BENCH_TRAIN_YEARS = 3  

# ============================================================
# 2) Descargar precios 
# ============================================================
def descargar_precios(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"{ticker}: descarga vacía. Revisa ticker o rango.")

    if "Close" in data.columns:
        close = data["Close"].dropna()
    else:
        close = data.iloc[:, 0].dropna()

    if isinstance(close, pd.Series):
        close = close.to_frame(name="Close")
    else:
        first_col = close.columns[0]
        if first_col != "Close":
            close = close.rename(columns={first_col: "Close"})
        close = close[["Close"]]

    if close.empty:
        raise ValueError(f"{ticker}: serie Close vacía tras limpiar NaNs.")
    return close

def sanity_check_close(close_df, ticker, max_abs_daily_ret=0.5):
    """
    max_abs_daily_ret: umbral de retornos extremos. Para ETFs grandes, >50% diario suele ser datos malos.
    """
    s = close_df["Close"]
    if s.isna().all():
        raise ValueError(f"{ticker}: todo NaN.")

    r = s.pct_change()
    mx = float(r.abs().max(skipna=True))
    if np.isfinite(mx) and mx > max_abs_daily_ret:
        print(f"WARNING {ticker}: retorno diario extremo {mx:.2%}. Posibles datos corruptos/splits.")


# ============================================================
# 3) HMM gaussiano multivariante con Baum–Welch (entrenamiento)
#    Observaciones: [r_t, |r_t|] --> [RETURNS, VOLATILITY]
# ============================================================
def entrenar_hmm_probabilistico(observaciones, n_states=N_ESTADOS, max_iter=50,
                                random_state=1, tol=1e-6):
    rng = np.random.default_rng(random_state)

    obs = np.asarray(observaciones.values, dtype=float)
    T = obs.shape[0]
    d = obs.shape[1]
    n_states = int(n_states)

    if T < n_states:
        raise ValueError("No hay suficientes observaciones para el número de estados elegido.")

    indices = rng.choice(T, size=n_states, replace=False)
    medias_full = obs[indices, :].copy()

    if T > 1:
        global_cov = np.cov(obs.T)
    else:
        global_cov = np.eye(d)
    if np.ndim(global_cov) == 0:
        global_cov = np.eye(d) * float(global_cov)
    global_cov = np.asarray(global_cov, dtype=float)
    global_cov += np.eye(d) * 1e-6

    covs = np.array([global_cov.copy() for _ in range(n_states)], dtype=float)

    pi = np.full(n_states, 1.0 / n_states, dtype=float)
    A = np.full((n_states, n_states), 1.0 / n_states, dtype=float)

    loglik_prev = -np.inf

    for _ in range(max_iter):
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

        xi = np.zeros((T - 1, n_states, n_states), dtype=float)
        for t in range(T - 1):
            numer = alpha[t, :, None] * A * (B[t + 1, :] * beta[t + 1, :])[None, :]
            denom = numer.sum()
            if denom == 0.0:
                denom = 1e-300
            xi[t, :, :] = numer / denom

        loglik = -np.sum(np.log(c))

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
# 3b) Filtrado + suavizado con parámetros fijos 
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
# 3c) Construir HMM walk-forward (sin look-ahead de parámetros)
# ============================================================
def construir_hmm_walkforward(observaciones, n_states=N_ESTADOS,
                              max_iter=MAX_ITER, random_state=42):
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

        _, _, medias_full, covs, A, pi = entrenar_hmm_probabilistico(
            obs_train, n_states=n_states, max_iter=max_iter, random_state=random_state
        )

        posterior_filt_test, posterior_smooth_test = filtrar_y_suavizar_hmm_dado_parametros(
            obs_test, medias_full, covs, A, pi
        )

        posterior_filt_list.append(posterior_filt_test)
        posterior_smooth_list.append(posterior_smooth_test)

        medias_r = medias_full[:, 0]
        sigmas_r = np.sqrt(covs[:, 0, 0])

        idx_test = obs_test.index
        cols = [f"state_{k}" for k in range(n_states)]

        medias_df_list.append(pd.DataFrame(np.tile(medias_r, (len(idx_test), 1)), index=idx_test, columns=cols))
        sigmas_df_list.append(pd.DataFrame(np.tile(sigmas_r, (len(idx_test), 1)), index=idx_test, columns=cols))

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
# 4) Métricas
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
# 5) Estrategia HMM -> Sharpe esperado + tendencia + vol targeting
# ============================================================
def estrategia_hmm_prob_regime_trend(precios, posterior_filt, medias_df, sigmas_df,
                                     pred_weights, freq_anual=252):
    close = precios["Close"]
    close_safe = precios["Close_safe"]

    ret_risk = close.pct_change().fillna(0.0)
    ret_safe = close_safe.pct_change().fillna(0.0)

    returns = close.pct_change().fillna(0.0)

    posterior = posterior_filt.reindex(returns.index).ffill()
    medias_df = medias_df.reindex(returns.index).ffill()
    sigmas_df = sigmas_df.reindex(returns.index).ffill()
    pred_weights = pred_weights.reindex(returns.index).ffill()

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

    # Vol targeting (sobre cartera preliminar)
    w_risk0 = position_pre_vol.clip(0.0, MAX_LEVERAGE)
    w_safe0 = (1.0 - w_risk0).clip(0.0, 1.0)
    strat_pre = w_risk0 * ret_risk + w_safe0 * ret_safe

    rolling_vol = strat_pre.rolling(ROLLING_VOL_DIAS).std() * np.sqrt(freq_anual)
    vol_scalar = TARGET_VOL_ANUAL / (rolling_vol + 1e-8)
    vol_scalar = vol_scalar.clip(0.0, MAX_LEVERAGE)
    vol_scalar = vol_scalar.shift(1).fillna(1.0)

    position = (position_pre_vol * vol_scalar).clip(0.0, MAX_LEVERAGE)

    # Pesos "raw" (diarios)
    w_risk_raw = position.clip(0.0, MAX_LEVERAGE)

    # ============================================================
    # REBALANCEO CADA N DÍAS 
    # ============================================================
    REBALANCE_EVERY = 5  # 5 = semanal aprox, 21 = mensual aprox
    idx = np.arange(len(w_risk_raw))
    reb_mask = (idx % REBALANCE_EVERY) == 0
    w_risk_reb = w_risk_raw.where(reb_mask).ffill().fillna(w_risk_raw.iloc[0])

    # ============================================================
    # CAMBIO 2) BANDA / ZONA MUERTA (evita microajustes)
    # ============================================================
    BAND = 0.05  # 5% de banda
    w_prev = w_risk_reb.shift(1).fillna(w_risk_reb.iloc[0])
    w_risk = w_risk_reb.copy()
    w_risk = pd.Series(
        np.where((w_risk - w_prev).abs() < BAND, w_prev, w_risk),
        index=w_risk.index
    ).clip(0.0, MAX_LEVERAGE)

    w_safe = (1.0 - w_risk).clip(0.0, 1.0)

    # ============================================================
    # EJECUCIÓN:
    # ============================================================
    w_risk_exec = w_risk.shift(1).fillna(w_risk.iloc[0])
    w_safe_exec = (1.0 - w_risk_exec).clip(0.0, 1.0)

    # ------------------------------------------------------------
    # Returns BRUTOS con pesos ejecutables
    # ------------------------------------------------------------
    strat_returns_gross = w_risk_exec * ret_risk + w_safe_exec * ret_safe

    # ------------------------------------------------------------
    # COSTES DE TRANSACCIÓN sobre cambios de peso ejecutable
    # (cartera 2-activos => turnover = 2*|Δw_risk|)
    # ------------------------------------------------------------
    TC_BPS = 15
    tc = TC_BPS / 10000.0

    dw = w_risk_exec.diff().abs().fillna(0.0)
    turnover = 2.0 * dw
    cost = tc * turnover

    # Returns NETOS 
    strat_returns = strat_returns_gross - cost

    # Métricas sobre NETO
    metricas, equity, drawdown = calcular_metricas(strat_returns, freq_anual=freq_anual)

    df_resultados = pd.DataFrame({
        "precio": close,
        "ret_risk": ret_risk,
        "ret_safe": ret_safe,
        "w_risk_raw": w_risk_raw,
        "w_risk": w_risk,                
        "w_risk_exec": w_risk_exec,       
        "w_safe_exec": w_safe_exec,
        "turnover": turnover,
        "cost": cost,
        "strat_returns_gross": strat_returns_gross,
        "strat_returns": strat_returns,   
        "equity": equity,
        "drawdown": drawdown
    })

    return df_resultados, metricas

# ============================================================
# 6) Benchmark helper
# ============================================================
def find_static_weights_for_target_vol(ret_risk, ret_safe, target_vol, freq_anual=252):
    vol_r = ret_risk.std() * np.sqrt(freq_anual)
    vol_s = ret_safe.std() * np.sqrt(freq_anual)
    rho = ret_risk.corr(ret_safe)

    def port_vol(w):
        var = (
            (w**2) * vol_r**2 +
            ((1-w)**2) * vol_s**2 +
            2*w*(1-w)*rho*vol_r*vol_s
        )
        return np.sqrt(var)

    ws = np.linspace(0, 1, 2001)
    vols = np.array([port_vol(w) for w in ws])

    idx = np.argmin(np.abs(vols - target_vol))
    w_risk = float(ws[idx])
    w_safe = float(1.0 - w_risk)

    return w_risk, w_safe, float(vols[idx])


# ============================================================
# 6b) Benchmark  
# ============================================================
def build_static_mix_ex_ante(ret_risk, ret_safe, strat_ret, freq_anual=252, train_years=3):
    # Alinear índices
    ret_risk = ret_risk.dropna()
    ret_safe = ret_safe.dropna()
    strat_ret = strat_ret.dropna()

    common = ret_risk.index.intersection(ret_safe.index).intersection(strat_ret.index)
    if len(common) < 60:
        raise ValueError("Benchmark ex-ante: no hay suficientes datos comunes entre series.")

    ret_risk = ret_risk.loc[common]
    ret_safe = ret_safe.loc[common]
    strat_ret = strat_ret.loc[common]

    idx0 = common.min()
    cutoff = idx0 + pd.DateOffset(years=train_years)

    r_train = ret_risk.loc[:cutoff]
    s_train = ret_safe.loc[:cutoff]
    strat_train = strat_ret.loc[:cutoff]

    if len(r_train) < 60 or len(s_train) < 60 or len(strat_train) < 60:
        raise ValueError("Benchmark ex-ante: periodo de entrenamiento demasiado corto.")

    # Target vol = vol de la estrategia en TRAIN (ex-ante)
    target_vol = strat_train.std() * np.sqrt(freq_anual)

    # Encuentra pesos (usando SOLO TRAIN) para igualar ese riesgo
    w_r, w_s, vol_b = find_static_weights_for_target_vol(r_train, s_train, target_vol, freq_anual=freq_anual)

    # Mix fijo aplicado a todo el periodo
    mix = (w_r * ret_risk + w_s * ret_safe).fillna(0.0)

    return mix, w_r, w_s, vol_b, target_vol
    
# ============================================================
# 7) Ejecución y gráficas
# ============================================================
def run_pair_and_plot(hmm_ticker, other_ticker, start, end):
    print(f"\n=====  {hmm_ticker} & {other_ticker} =====")

    prices_risk = descargar_precios(hmm_ticker, start, end)
    prices_other = descargar_precios(other_ticker, start, end)

    sanity_check_close(prices_risk, hmm_ticker)
    sanity_check_close(prices_other, other_ticker)

    prices = prices_risk.copy()
    prices["Close_safe"] = prices_other["Close"]
    prices = prices.dropna()

    r = prices["Close"].pct_change().dropna()
    obs = pd.DataFrame({"r": r, "v": r.abs()}, index=r.index)

    posterior_filt, posterior_smooth, medias_df, sigmas_df, pred_weights = construir_hmm_walkforward(
        obs, n_states=N_ESTADOS, max_iter=MAX_ITER, random_state=42
    )

    first_idx = posterior_filt.index.min()
    prices_sub = prices.loc[first_idx:]

    df_resultados, metricas = estrategia_hmm_prob_regime_trend(
        prices_sub, posterior_filt, medias_df, sigmas_df, pred_weights, freq_anual=FREQ_ANUAL
    )

    # returns
    ret_risk = df_resultados["ret_risk"].fillna(0.0)
    ret_other = df_resultados["ret_safe"].fillna(0.0)
    strat_ret = df_resultados["strat_returns"].fillna(0.0)
    
    # ============================================================
    # Regresión Lineal Multivariante
    # ============================================================

    df_alpha = pd.DataFrame({
        "strat": strat_ret,
        "gld": ret_risk,
        "other": ret_other
    }).dropna()

    Y = df_alpha["strat"]
    X = sm.add_constant(df_alpha[["gld", "other"]])

    # Newey–West para autocorrelación
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_daily = res.params["const"]
    alpha_ann = alpha_daily * FREQ_ANUAL

    print("\n--- LINEAR REGRESSION ---")
    print(f"Annualized alpha: {alpha_ann:.2%}")
    print(f"t-stat alpha: {res.tvalues['const']:.2f}")
    print(f"p-value alpha: {res.pvalues['const']:.4f}")


    # --------------------------------------------------------
    # BENCHMARK EX-ANTE 
    # --------------------------------------------------------
    bh_mix, w_r, w_o, vol_b, target_vol = build_static_mix_ex_ante(
        ret_risk, ret_other, strat_ret,
        freq_anual=FREQ_ANUAL, train_years=BENCH_TRAIN_YEARS
    )

    # equity curves
    eq_strat = (1 + strat_ret).cumprod()
    eq_risk = (1 + ret_risk).cumprod()
    eq_other = (1 + ret_other).cumprod()
    eq_mix = (1 + bh_mix).cumprod()

    # align start and normalize to 1
    start_dt = max(eq_strat.first_valid_index(), eq_risk.first_valid_index(),
                   eq_other.first_valid_index(), eq_mix.first_valid_index())

    eq_strat = eq_strat.loc[start_dt:] / eq_strat.loc[start_dt]
    eq_risk = eq_risk.loc[start_dt:] / eq_risk.loc[start_dt]
    eq_other = eq_other.loc[start_dt:] / eq_other.loc[start_dt]
    eq_mix = eq_mix.loc[start_dt:] / eq_mix.loc[start_dt]

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(eq_strat.index, eq_strat.values, label="STRATEGY")
    plt.plot(eq_risk.index, eq_risk.values, label=f"BH {hmm_ticker}")
    plt.plot(eq_other.index, eq_other.values, label=f"BH {other_ticker}")
    plt.plot(
        eq_mix.index, eq_mix.values,
        label=f"BH MIX ex-ante (train {BENCH_TRAIN_YEARS}y, target vol=strat train, w{hmm_ticker}={w_r:.2f})"
    )

    plt.title(f"Equity curves: Strategy vs Buy&Hold ({hmm_ticker} / {other_ticker} / Mix ex-ante)")
    plt.xlabel("Fecha")
    plt.ylabel("Equity (base = 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# ============================================================
# 8) MAIN
# ============================================================

def main():
    print(f"Backtest multi-pairs: {HMM_TICKER} vs {len(PAIRS)} activos")
    print(f"Rango: {START_DATE} -> {END_DATE}")
    print(f"Benchmark ex-ante: calibra con primeros {BENCH_TRAIN_YEARS} años (vol de estrategia en train).")
    print (f"Training HMM model")
    for i in range (11):
        print (f"Training completed: {10*i}%")
        time.sleep(1)
    for other in PAIRS:
        try:
            run_pair_and_plot(HMM_TICKER, other, START_DATE, END_DATE)
        except Exception as e:
            print(f"ERROR en {HMM_TICKER} vs {other}: {e}")


if __name__ == "__main__":
    main()
