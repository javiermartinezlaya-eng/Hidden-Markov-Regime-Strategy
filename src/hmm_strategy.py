import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import os
from src.analysis.signal_diagnostics import (
    analizar_lag_exp_sharpe,
    plot_sigmoid_and_nextday,
    plot_return_hist_with_signal_means,
)

# ============================================================
# 1. Parámetros
# ============================================================
TICKER = "QQQ"
START_DATE = "2010-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

N_ESTADOS = 3
MAX_ITER = 50
FREQ_ANUAL = 252  # bolsa USA

EMA_FAST = 25
EMA_SLOW = 140

REGIME_CONF_DAYS = 1  

# --------- Parámetros específicos de esta versión ----------
# Sharpe esperado diario -> posición
EXP_SHARPE_SCALE = 0.30     # escala típica: Sharpe ~0.30 -> saturación de sigmoide
EXP_SHARPE_CUTOFF = 0.05    # por debajo de 0.05 -> posición casi 0

# Volatility targeting
TARGET_VOL_ANUAL = 0.15     # 14% volatilidad objetivo
ROLLING_VOL_DIAS = 20       # ventana para estimar vol
MAX_LEVERAGE = 1.2          # apalancamiento máximo

# Filtro de tendencia: peso mínimo/máximo
TREND_MIN_WEIGHT = 0.0      # sin tendencia -> multiplicador mínimo
TREND_MAX_WEIGHT = 1.0      # con tendencia -> multiplicador máximo
# -----------------------------------------------------------


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
                                random_state=91, tol=1e-6):
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
    if global_cov.shape == ():
        global_cov = np.eye(d) * float(global_cov)
    global_cov = np.asarray(global_cov, dtype=float)
    # regularización
    global_cov += np.eye(d) * 1e-6

    covs = np.array([global_cov.copy() for _ in range(n_states)], dtype=float)

    # pi y A uniformes
    pi = np.full(n_states, 1.0 / n_states, dtype=float)
    A = np.full((n_states, n_states), 1.0 / n_states, dtype=float)

    loglik_prev = -np.inf

    for it in range(max_iter):
        # ======================
        # E-step: calcular B, alpha, beta, gamma, xi
        # ======================
        B = np.zeros((T, n_states), dtype=float)

        for k in range(n_states):
            mu_k = medias_full[k]
            Sigma_k = covs[k]

            # regularización por si acaso
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

        gamma_sum = gamma.sum(axis=0)  # (n_states,)

        for k in range(n_states):
            if gamma_sum[k] == 0.0:
                medias_full[k] = np.zeros(d)
                covs[k] = np.eye(d)
            else:
                w = gamma[:, k][:, None]  # (T,1)
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
        beta[t] /= c[t + 1]

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
#       Entrenamos por años: para el año Y se entrena con datos < Y
#       Observaciones: DataFrame con columnas ["r", "v"]
# ============================================================
def construir_hmm_walkforward(observaciones, n_states=N_ESTADOS,
                              max_iter=MAX_ITER, random_state=42):
    """
    Devuelve:
    - posterior_filt_all  : probabilidades filtradas walk-forward (P(s_t | obs_1..t))
    - posterior_smooth_all: probabilidades suavizadas (por año, sólo análisis)
    - medias_df_all       : DataFrame (fecha x estado) con medias de retornos por estado
    - sigmas_df_all       : igual para sigma de retornos
    - pred_weights_all    : PREDICTIVO P(s_t | info hasta t-1) para cada fecha (sin look-ahead)
    """
    years = sorted(observaciones.index.year.unique())
    if len(years) < 2:
        raise ValueError("Se necesita al menos 2 años de datos para hacer walk-forward.")

    posterior_filt_list = []
    posterior_smooth_list = []
    medias_df_list = []
    sigmas_df_list = []
    pred_weights_list = []  # NUEVO: pesos predictivos P(s_t | info hasta t-1)

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

        # Entrenamos sólo con datos < year (no look-ahead)
        _, _, medias_full, covs, A, pi = entrenar_hmm_probabilistico(
            obs_train, n_states=n_states, max_iter=max_iter, random_state=random_state
        )

        # Filtramos en el año test usando esos parámetros (sin reentrenar)
        posterior_filt_test, posterior_smooth_test = filtrar_y_suavizar_hmm_dado_parametros(
            obs_test, medias_full, covs, A, pi
        )

        posterior_filt_list.append(posterior_filt_test)
        posterior_smooth_list.append(posterior_smooth_test)

        # Medias y sigmas SOLO del retorno (dimensión 0)
        medias_r = medias_full[:, 0]               # (n_states,)
        sigmas_r = np.sqrt(covs[:, 0, 0])          # (n_states,)

        idx_test = obs_test.index
        cols = [f"state_{k}" for k in range(n_states)]

        medias_df_list.append(
            pd.DataFrame(
                np.tile(medias_r, (len(idx_test), 1)),
                index=idx_test,
                columns=cols
            )
        )
        sigmas_df_list.append(
            pd.DataFrame(
                np.tile(sigmas_r, (len(idx_test), 1)),
                index=idx_test,
                columns=cols
            )
        )

        # -------------------------------------------------------
        # NUEVO: Pesos PREDICTIVOS P(s_t | info hasta t-1)
        # Para el primer día del año usamos pi (entrenado sin look-ahead).
        # Para t>0: P_pred(s_t) = P_filt(s_{t-1}) @ A
        # -------------------------------------------------------
        alpha_vals = posterior_filt_test.values  # P(s_t | obs_1..t)
        T_test, n_states_eff = alpha_vals.shape
        assert n_states_eff == n_states

        pred_weights = np.zeros_like(alpha_vals)
        pred_weights[0, :] = pi  # distribución inicial basada en entrenamiento (<year)

        for t in range(1, T_test):
            pred_weights[t, :] = alpha_vals[t-1, :] @ A

        pred_weights_df = pd.DataFrame(
            pred_weights,
            index=idx_test,
            columns=cols
        )
        pred_weights_list.append(pred_weights_df)

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
#    usando posterior FILTRADO walk-forward + medias/sigmas de retorno por fecha
#    y pesos PREDICTIVOS (sin look-ahead) para el Sharpe
# ============================================================
def estrategia_hmm_prob_regime_trend(precios, posterior_filt, medias_df, sigmas_df,
                                     pred_weights, freq_anual=252, fwd_horizon_days=1,
                                     use_cumulative_forward_return=True):
    close = precios["Close"]
    returns = close.pct_change().fillna(0.0)

    posterior = posterior_filt.reindex(returns.index).ffill()
    medias_df = medias_df.reindex(returns.index).ffill()
    sigmas_df = sigmas_df.reindex(returns.index).ffill()
    pred_weights = pred_weights.reindex(returns.index).ffill()  # P(s_t | info hasta t-1)

    # 1) Regímenes duros basados en posterior filtrado (P(s_t | obs_1..t))
    regimes_hard = posterior.values.argmax(axis=1)
    regimes_hard = pd.Series(regimes_hard, index=returns.index, name="regime_hard")

    medias_por_estado = returns.groupby(regimes_hard).mean().sort_index()
    print("\n=== Medias de retorno por estado HMM (duro, walk-forward multivariante) ===")
    for s in medias_por_estado.index:
        print(f"Estado {s}: media retorno diaria = {medias_por_estado[s]:.6f}")

    # 2) EMAs de tendencia (causal)
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()

    trend_flag = (ema_fast > ema_slow).astype(float)
    trend_flag_shifted = trend_flag.shift(1).fillna(0.0)

    # 3) Sharpe esperado por mezcla HMM (dimensión retorno) usando
    #    PREDICTIVO: P(s_t | info hasta t-1) -> sin look-ahead
    medias_arr = medias_df.values            # (T, n_states)
    sigmas_arr = sigmas_df.values            # (T, n_states)
    weights_arr = pred_weights.values        # (T, n_states) P(s_t | info hasta t-1)

    mu_mix = np.sum(weights_arr * medias_arr, axis=1)
    var_mix = np.sum(weights_arr * (sigmas_arr**2 + medias_arr**2), axis=1) - mu_mix**2
    var_mix = np.clip(var_mix, 1e-10, None)
    sigma_mix = np.sqrt(var_mix)

    exp_sharpe_daily = mu_mix / (sigma_mix + 1e-8)
    exp_sharpe_daily = pd.Series(exp_sharpe_daily, index=returns.index)

    # Sharpe esperado "usable" (forward): sólo suavizado (no shift)
    exp_sharpe_smooth = exp_sharpe_daily.ewm(span=5, adjust=False).mean()
    exp_sharpe_shifted = exp_sharpe_smooth.shift(1)
    # Sigmoide con escala EXP_SHARPE_SCALE
    k = 5.0
    x0 = EXP_SHARPE_CUTOFF
    scale = EXP_SHARPE_SCALE

    z = (exp_sharpe_shifted - x0) / scale
    base_position = 1.0 / (1.0 + np.exp(-k * z))
    base_position = pd.Series(base_position, index=returns.index)

    # 4) Filtro de tendencia
    trend_adjust = TREND_MIN_WEIGHT + (TREND_MAX_WEIGHT - TREND_MIN_WEIGHT) * trend_flag_shifted
    position_pre_vol = base_position * trend_adjust

    # 5) Volatility Targeting
    rolling_vol = returns.rolling(ROLLING_VOL_DIAS).std() * np.sqrt(freq_anual)
    vol_scalar = TARGET_VOL_ANUAL / (rolling_vol + 1e-8)
    vol_scalar = vol_scalar.clip(0.0, MAX_LEVERAGE)
    vol_scalar = vol_scalar.shift(1).fillna(1.0)

    position = (position_pre_vol * vol_scalar).clip(0.0, MAX_LEVERAGE)

    # 6) Retornos estrategia base HMM
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
        "exp_sharpe_shifted": exp_sharpe_shifted,   # Sharpe suavizado ex-ante
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

    # ------------------------------------------------------------
    # Correlación y regresión lineal simple:
    # RETORNO FUTURO del activo (no la estrategia):
    # forward_return_{t->t+h} = alpha + beta * exp_sharpe_shifted_t
    # ------------------------------------------------------------
    h = max(1, int(fwd_horizon_days))
    serie_x = df_resultados["exp_sharpe_shifted"]

    if use_cumulative_forward_return:
        # Retorno acumulado desde t hasta t+h (aprox h días de trading)
        forward_return = (1.0 + df_resultados["returns"]).rolling(h).apply(np.prod, raw=True).shift(-h) - 1.0
    else:
        # Retorno del día t+h (NO acumulado)
        forward_return = df_resultados["returns"].shift(-h)

    serie_y = forward_return

    mask = serie_x.notna() & serie_y.notna()

    if mask.sum() >= 2:
        x_vals = serie_x[mask].values
        y_vals = serie_y[mask].values

        # Correlación de Pearson (señal vs retorno futuro del activo)
        corr_exp_sharp_shifted_returns = np.corrcoef(x_vals, y_vals)[0, 1]

        # Regresión lineal simple: y = beta * x + alpha
        beta, alpha = np.polyfit(x_vals, y_vals, 1)

        # R^2
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
# 8. Main (sólo HMM, sin ML encima)
# ============================================================
def main():
    print(f"Descargando {TICKER}...")
    prices = descargar_precios(TICKER, START_DATE, END_DATE)

    # =========================================================
    # 1) Observaciones para HMM multivariante
    # =========================================================
    r = prices["Close"].pct_change().dropna()
    obs = pd.DataFrame(
        {"r": r, "v": r.abs()},
        index=r.index
    )

    print("\nConstruyendo HMM walk-forward multivariante (sin look-ahead de parámetros)…")
    posterior_filt, posterior_smooth, medias_df, sigmas_df, pred_weights = construir_hmm_walkforward(
        obs,
        n_states=N_ESTADOS,
        max_iter=MAX_ITER,
        random_state=42
    )

    # Subserie de precios desde la primera fecha con posterior
    first_idx = posterior_filt.index.min()
    prices_sub = prices.loc[first_idx:]

    # =========================================================
    # 2) Estrategia: HMM_expSharpe + trend + vol targeting
    # =========================================================
    print("\nAplicando estrategia HMM_expSharpe + trend + vol targeting (walk-forward multivariante)…")
    df_resultados, metricas, corr_exp_sharp_shifted_returns, alpha, beta, r2 = \
        estrategia_hmm_prob_regime_trend(
            prices_sub,
            posterior_filt,
            medias_df,
            sigmas_df,
            pred_weights,
            freq_anual=FREQ_ANUAL
        )

    print(f"\n=== Métricas Estrategia base HMM_expSharpe+Trend+VolTarget sobre {TICKER} ===")
    for k, v in metricas.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    # =========================================================
    # 3) Correlación y regresión (diagnóstico numérico)
    # =========================================================
    print("\n=== Correlación entre exp_sharpe_shifted y forward_return del activo ===")
    try:
        print(f"Coeficiente de correlación de Pearson: {float(corr_exp_sharp_shifted_returns):.4f}")
    except Exception:
        print(f"Coeficiente de correlación de Pearson: {corr_exp_sharp_shifted_returns}")

    print("\n=== Regresión lineal: forward_return = alpha + beta * exp_sharpe_shifted ===")
    print(f"alpha (intercepto): {alpha:.8f}")
    print(f"beta  (pendiente) : {beta:.8f}")
    print(f"R^2               : {r2:.4f}")

    # =========================================================
    # 4) Diagnósticos de señal (exporta PNGs)
    # =========================================================
    FIG_DIR = "reports/figures"
    os.makedirs(FIG_DIR, exist_ok=True)

    # 4.1 Lag sweep (se usa también para Excel)
    df_lags = analizar_lag_exp_sharpe(
        df_resultados,
        max_lag=20,
        save_path=os.path.join(FIG_DIR, "lag_sweep.png"),
    )

    # 4.2 Sigmoide vs retorno siguiente (3 figuras)
    plot_sigmoid_and_nextday(
        df_resultados,
        cutoff=EXP_SHARPE_CUTOFF,
        save_path_prefix=os.path.join(FIG_DIR, "sigmoid_nextday"),
    )

    # 4.3 Histograma retornos + medias de señal
    plot_return_hist_with_signal_means(
        df_resultados,
        bin_width=0.0025,
        r_min=-0.10,
        r_max=0.10,
        save_path=os.path.join(FIG_DIR, "return_hist_signal_means.png"),
    )

    # =========================================================
    # 5) Métricas anuales, equity y heatmap mensual
    # =========================================================
    annual_base = (
        df_resultados["strat_returns"]
        .groupby(df_resultados.index.year)
        .apply(lambda x: (1 + x).prod() - 1)
    )
    df_annual = pd.DataFrame({"annual_base": annual_base})

    print("\n=== Rentabilidad anual (base) ===")
    print(df_annual)

    bh_returns = df_resultados["precio"].pct_change().fillna(0)
    bh_equity = (1 + bh_returns).cumprod()

    eq_curve = pd.DataFrame({
        "precio": df_resultados["precio"],
        "equity_buyhold": bh_equity,
        "equity_base": df_resultados["equity"],
    })

    df_heat = df_resultados.copy()
    df_heat["year"] = df_heat.index.year
    df_heat["month"] = df_heat.index.month

    heatmap_base = (
        df_heat
        .groupby(["year", "month"])["strat_returns"]
        .apply(lambda x: (1 + x).prod() - 1)
        .unstack()
    )

    # =========================================================
    # 6) Exportar a Excel (base)
    # =========================================================
    file_name = (
        f"hmm_{TICKER}_{N_ESTADOS}estados_"
        f"expSharpe_trend_voltarget_walkforward_multivar_BASE.xlsx"
    )

    with pd.ExcelWriter(file_name) as writer:
        df_resultados.to_excel(writer, sheet_name="Resultados_diarios_base")
        df_annual.to_excel(writer, sheet_name="Rentabilidad_anual_base")
        eq_curve.to_excel(writer, sheet_name="Equity_curve")
        heatmap_base.to_excel(writer, sheet_name="Heatmap_mensual_base")
        posterior_smooth.to_excel(writer, sheet_name="Posterior_suavizado_walkforward")
        pred_weights.to_excel(writer, sheet_name="Pred_weights_walkforward")
        df_lags.to_excel(writer, sheet_name="Lag_exp_sharpe_vs_strat")

    print(f"\nExcel guardado como: {file_name}")

    # =========================================================
    # 7) Señal sugerida (explicada)
    # =========================================================
    last_dt = df_resultados.index[-1]
    row = df_resultados.iloc[-1]

    print("\n=== Señal sugerida por el modelo (última fecha disponible) ===")
    print(f"Fecha señal: {last_dt.date()}")

    print("\n--- Filtro 1: HMM + sigmoide ---")
    print(f"Sharpe esperado suavizado (exp_sharpe_shifted): {row['exp_sharpe_shifted']:.4f}")
    print(f"Base position (sigmoide): {row['base_position_sigmoid']:.4f}")

    print("\n--- Filtro 2: Tendencia ---")
    print(f"EMA fast > EMA slow (trend_flag): {int(row['trend_flag'])}")
    print(f"Trend adjust aplicado: {row['trend_adjust']:.2f}")

    print("\n--- Filtro 3: Volatility targeting ---")
    print(f"Vol anual estimada (rolling): {row['rolling_vol_anual']:.4f}")
    print(f"Vol scalar aplicado: {row['vol_scalar']:.4f}")

    print("\n=== RESULTADO FINAL ===")
    print(f"Posición sugerida por el modelo: {row['position']:.4f}")

if __name__ == "__main__":
    main()
