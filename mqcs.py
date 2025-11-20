"""
MQCS(τ) quantile-cointegration test  (MIT licence)
"""
import numpy as np
import statsmodels.api as sm
from arch.bootstrap import CircularBlockBootstrap

def _zscore(x):
    return (x - x.mean()) / x.std()

def mqcs(y, x, tau, h=None, B=1000, block_size=None, seed=42):
    n = len(y)
    h = int(n**(1/5)) if h is None else h
    block_size = int(n**(1/3)) if block_size is None else block_size
    X = sm.add_constant(x.reshape(-1, 1))
    mod = sm.QuantReg(y, X)
    beta = mod.fit(q=tau).params
    uhat = y - X @ beta
    psi = tau - (uhat < 0)
    lrv = sm.stats.sandwich_covariance.cov_nw(psi, h)
    S = np.cumsum(psi) / np.sqrt(n) / np.sqrt(lrv)
    stat = np.max(np.abs(S))
    rng = np.random.default_rng(seed)
    bs_stats = []
    cbb = CircularBlockBootstrap(block_size, x, y, random_state=rng)
    for _, bx, by in cbb.bootstrap(B):
        bX = sm.add_constant(bx.reshape(-1, 1))
        bmod = sm.QuantReg(by, bX).fit(q=tau)
        bu = by - bX @ bmod.params
        bpsi = tau - (bu < 0)
        blrv = sm.stats.sandwich_covariance.cov_nw(bpsi, h)
        bS = np.cumsum(bpsi) / np.sqrt(len(bpsi)) / np.sqrt(blrv)
        bs_stats.append(np.max(np.abs(bS)))
    pval = 1 - np.mean(np.array(bs_stats) <= stat)
    return {'stat': stat, 'pval': pval}
