#!/usr/bin/env python3
# CoVaR Framework — Darbyshire Ch.14 Implementation
# 18 risk factors: 13 SOFR OIS tenors + 5 LatAm FX
# Three covariance estimators: equal-weight, EWMA, RIE
# Outputs: CoVaR multiplier, 99% VaR, single instrument VaR min, VaR allocation

# ============================================================
# CELL 1: IMPORTS AND CONFIG
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import norm
from copy import deepcopy

# config
CONFIDENCE = 0.99
Z_ALPHA = norm.ppf(CONFIDENCE)  # 2.326
EWMA_LAMBDA = 0.94
RATE_TENORS = ['1mo','2mo','3mo','4mo','6mo','1y','2y','3y','5y','7y','10y','20y','30y']
FX_PAIRS = ['USDBRL','USDCOP','USDCLP','USDPEN','USDMXN', 'USDEUR'	,'USDGBP','USDJPY',	'USDAUD','USDCHF']
N_RATES = len(RATE_TENORS)
N_FX = len(FX_PAIRS)
N_TOTAL = N_RATES + N_FX
LABELS = RATE_TENORS + FX_PAIRS

# ============================================================
# CELL 2: FUNCTIONS
# ============================================================

def load_and_transform(filepath):
    # read csv, first col is date
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    # first column is date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    # rename columns to standard labels
    # assume order: 13 rate cols then 5 fx cols
    data_cols = [c for c in df.columns]
    if len(data_cols) != N_TOTAL:
        raise ValueError(f'expected {N_TOTAL} data columns, got {len(data_cols)}')
    df.columns = LABELS
    # compute daily changes
    # rates: absolute first difference x 100 (levels are in %, so diff is in pp, x100 = bps)
    rates_levels = df[RATE_TENORS]
    rates_chg = rates_levels.diff() * 100  # bps per day
    # fx: relative first difference (% change per day)
    fx_levels = df[FX_PAIRS]
    fx_chg = fx_levels.pct_change() * 100  # % per day
    # combine and drop first row (NaN from diff)
    changes = pd.concat([rates_chg, fx_chg], axis=1).dropna()
    return df, changes

def cov_equal_weight(changes):
    # sample covariance matrix (unbiased, ddof=1)
    return changes.cov().values

def cov_ewma(changes, lam=EWMA_LAMBDA):
    # exponentially weighted covariance matrix
    T, N = changes.shape
    X = changes.values
    Q = np.zeros((N, N))
    # initialise with first observation outer product
    Q = np.outer(X[0], X[0])
    for t in range(1, T):
        Q = lam * Q + (1 - lam) * np.outer(X[t], X[t])
    return Q

def cov_rie(changes):
    # rotationally invariant estimator (Bun, Bouchaud, Potters 2016)
    T, N = changes.shape
    q = N / T
    # sample correlation matrix from changes
    X = changes.values
    # standardise: demean and normalise by std
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1e-10
    Z = (X - mu) / sigma
    # sample correlation matrix
    E = Z.T @ Z / T
    # eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(E)
    # sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # clip negative eigenvalues (numerical)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    # stieltjes transform: s(z) = (1/N) * tr((zI - E)^{-1})
    # for z_k = lambda_k - i*eta
    eta = 1.0 / np.sqrt(N)
    cleaned_eigenvalues = np.zeros(N)
    for k in range(N):
        z_k = eigenvalues[k] - 1j * eta
        # s(z_k) = (1/N) * sum_j (1 / (z_k - lambda_j))
        s_zk = np.mean(1.0 / (z_k - eigenvalues))
        # RIE formula: xi_k = lambda_k / |1 - q + q*z_k*s(z_k)|^2
        denom = abs(1 - q + q * z_k * s_zk) ** 2
        xi_k = eigenvalues[k] / denom
        cleaned_eigenvalues[k] = xi_k
    # debiasing correction (eq 10 in paper)
    # xi_hat_k = xi_k * max(1, lambda_k) — simplified heuristic
    # for our case with small q, correction is minor
    mp_upper = (1 + np.sqrt(q)) ** 2
    for k in range(N):
        correction = max(1.0, eigenvalues[k] / mp_upper)
        cleaned_eigenvalues[k] = cleaned_eigenvalues[k] * correction
    # ensure trace preservation: cleaned eigenvalues should sum to N
    cleaned_eigenvalues = cleaned_eigenvalues * (N / cleaned_eigenvalues.sum())
    # reconstruct cleaned correlation matrix
    C_clean = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
    # force unit diagonal
    d = np.sqrt(np.diag(C_clean))
    d[d == 0] = 1e-10
    C_clean = C_clean / np.outer(d, d)
    # convert back to covariance matrix using original volatilities
    vol = changes.std(ddof=1).values
    Q_rie = np.outer(vol, vol) * C_clean
    return Q_rie

def covar_multiplier(S, Q):
    # c = sqrt(S^T Q S)
    return np.sqrt(S @ Q @ S)

def covar_var(S, Q, z=Z_ALPHA):
    # VaR = z * c
    c = covar_multiplier(S, Q)
    return z * c

def single_instrument_var_min(S, Q):
    # S_trade_i = -(c / q_ii) * (QS)_i / c = -(QS)_i / q_ii
    # per Darbyshire eq 14.3 derivation
    c = covar_multiplier(S, Q)
    QS = Q @ S
    diag_Q = np.diag(Q)
    # S_trade: the position in each bucket that minimises VaR
    # from a single instrument perspective
    S_trade = -c * QS / (diag_Q * c)  # simplifies to -QS / diag_Q
    S_trade = -QS / diag_Q
    # delta VaR from each trade (approximate)
    # VaR reduction per unit = (QS)_i / c, the marginal VaR
    marginal_var = Z_ALPHA * QS / c
    return S_trade, marginal_var

def var_allocation(S, Q):
    # Euler allocation: C_alloc_i = S_i * (QS)_i / c
    c = covar_multiplier(S, Q)
    QS = Q @ S
    component_var = Z_ALPHA * S * QS / c
    # these sum to total VaR
    return component_var

def historical_var(changes, S, confidence=CONFIDENCE):
    # historical simulation: compute daily P&L for each day
    # P&L_t = S . changes_t (element-wise product summed)
    pnl = changes.values @ S
    # VaR = negative quantile at (1 - confidence)
    var_hist = -np.percentile(pnl, (1 - confidence) * 100)
    return var_hist, pnl

def run_comparison(S, changes, label='Portfolio'):
    # compute all three covariance matrices
    Q_ew = cov_equal_weight(changes)
    Q_ewma = cov_ewma(changes)
    Q_rie = cov_rie(changes)
    # historical VaR
    h_var, pnl = historical_var(changes, S)
    print(f'\n{"="*70}')
    print(f' CoVaR ANALYSIS: {label}')
    print(f'{"="*70}')
    print(f' Confidence: {CONFIDENCE*100:.0f}% | z = {Z_ALPHA:.3f}')
    print(f' Risk factors: {N_TOTAL} | Observations: {len(changes)}')
    print(f' q = N/T = {N_TOTAL/len(changes):.4f}')
    print(f'{"="*70}\n')
    results = {}
    for name, Q in [('Equal Weight', Q_ew), ('EWMA 0.94', Q_ewma), ('RIE Cleaned', Q_rie)]:
        c = covar_multiplier(S, Q)
        var = covar_var(S, Q)
        s_trade, mvar = single_instrument_var_min(S, Q)
        alloc = var_allocation(S, Q)
        results[name] = {
            'Q': Q, 'c': c, 'var': var,
            's_trade': s_trade, 'marginal_var': mvar, 'allocation': alloc
        }
        print(f'--- {name} ---')
        print(f'  CoVaR multiplier (c):  {c:>12,.2f}')
        print(f'  99% VaR:               {var:>12,.2f}')
        print(f'')
    print(f'--- Historical VaR (99%) ---')
    print(f'  Historical VaR:        {h_var:>12,.2f}')
    print(f'')
    # var minimisation table
    print(f'\n{"="*70}')
    print(f' SINGLE INSTRUMENT VAR MINIMISATION')
    print(f'{"="*70}')
    header = f'{"Bucket":<10} {"S (current)":<12} '
    for name in results:
        header += f'{"S_min("+name[:6]+")":<16} '
    print(header)
    print('-' * len(header))
    for i, lab in enumerate(LABELS):
        row = f'{lab:<10} {S[i]:>11,.2f} '
        for name in results:
            row += f'{results[name]["s_trade"][i]:>15,.2f} '
        print(row)
    # allocation table
    print(f'\n{"="*70}')
    print(f' VAR ALLOCATION (99%)')
    print(f'{"="*70}')
    header = f'{"Bucket":<10} '
    for name in results:
        header += f'{"Alloc("+name[:6]+")":<16} '
    header += f'{"% of VaR":<12}'
    print(header)
    print('-' * len(header))
    for i, lab in enumerate(LABELS):
        row = f'{lab:<10} '
        for name in results:
            row += f'{results[name]["allocation"][i]:>15,.2f} '
        # pct using first method
        first_name = list(results.keys())[0]
        pct = results[first_name]['allocation'][i] / results[first_name]['var'] * 100
        row += f'{pct:>10.1f}%'
        print(row)
    print('-' * 70)
    row_total = f'{"TOTAL":<10} '
    for name in results:
        row_total += f'{results[name]["allocation"].sum():>15,.2f} '
    print(row_total)
    return results, h_var, pnl

def export_for_excel(Q, diag_Q, labels, filename_prefix):
    # export Q matrix and diag(Q) as CSVs for Excel paste
    df_Q = pd.DataFrame(Q, index=labels, columns=labels)
    df_Q.to_csv(f'C:/Users/user/Downloads/{filename_prefix}_Q.csv')
    df_diag = pd.DataFrame({'bucket': labels, 'diag_Q': diag_Q})
    df_diag.to_csv(f'C:/Users/user/Downloads/{filename_prefix}_diag_Q.csv', index=False)
    print(f'  exported: {filename_prefix}_Q.csv, {filename_prefix}_diag_Q.csv')


# ============================================================
# CELL 3: MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    # --- load data ---
    filepath = 'C:/Users/user/Downloads/riskfactors.csv'
    try:

        raw, changes = load_and_transform(filepath)
        print(f'loaded {len(raw)} rows, {len(changes)} daily changes')
        print(f'date range: {raw.index[0]} to {raw.index[-1]}')
        print(f'columns: {list(raw.columns)}')
    except FileNotFoundError:
        print('riskfactors.csv not found — generating synthetic data for testing')
        np.random.seed(42)
        T = 547
        dates = pd.bdate_range('2023-01-02', periods=T)
        # synthetic SOFR OIS levels (starting ~5%, slight trend + noise)
        rate_data = {}
        base_rates = [5.30, 5.28, 5.25, 5.22, 5.15, 4.80, 4.20, 3.90, 3.70, 3.60, 3.55, 3.50, 3.48]
        for i, tenor in enumerate(RATE_TENORS):
            level = base_rates[i]
            noise = np.cumsum(np.random.normal(0, 0.02, T))
            rate_data[tenor] = level + noise
        # synthetic LatAm FX
        fx_data = {}
        base_fx = {'USDBRL': 5.00, 'USDCOP': 4000, 'USDCLP': 850, 'USDPEN': 3.75, 'USDMXN': 17.5}
        for pair, base in base_fx.items():
            noise = np.cumsum(np.random.normal(0, base * 0.005, T))
            fx_data[pair] = base + noise
        raw = pd.DataFrame({**rate_data, **fx_data}, index=dates)
        # compute changes
        rates_chg = raw[RATE_TENORS].diff() * 100
        fx_chg = raw[FX_PAIRS].pct_change() * 100
        changes = pd.concat([rates_chg, fx_chg], axis=1).dropna()
        print(f'generated synthetic data: {len(changes)} daily changes')

    # --- example risk position (S vector) ---
    # user should replace with actual DV01s and FX deltas from Excel
    # DV01s in USD (negative = paying fixed / short duration)
    # FX deltas in USD (positive = long USD vs LatAm)
    S = np.array([
        # SOFR OIS DV01s (USD per bp)
        0, 0, 0, 0, 0,       # 1M-6M: no short-end position
        500, 1000, 2000, 3000, 2000,  # 1Y-7Y: long duration
        -5000, -2000, -1000,  # 10Y-30Y: short duration
        # LatAm FX deltas (USD per 1% move)
        -10000, 5000, 0, -20000, 8000,1,1,1,1,1  # BRL, COP, CLP, PEN, MXN
    ])
    print(f'\nrisk position S:')
    for i, lab in enumerate(LABELS):
        print(f'  {lab:<10} {S[i]:>12,.0f}')

    # --- run comparison ---
    results, h_var, pnl = run_comparison(S, changes, label='LatAm Rates + FX Portfolio')

    # --- export Q matrices for Excel ---
    print(f'\n{"="*70}')
    print(f' EXPORTING FOR EXCEL')
    print(f'{"="*70}')
    for name, short in [('Equal Weight', 'ew'), ('EWMA 0.94', 'ewma'), ('RIE Cleaned', 'rie')]:
        Q = results[name]['Q']
        export_for_excel(Q, np.diag(Q), LABELS, f'covar_{short}')
    print(f'\nDone. Paste Q and diag(Q) into Excel alongside live S vector.')
    print(f'CoVaR multiplier = SQRT(MMULT(TRANSPOSE(S), MMULT(Q, S)))')
    print(f'VaR 99% = {Z_ALPHA:.3f} * CoVaR multiplier')
