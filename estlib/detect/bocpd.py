# estlib/detect/bocpd.py
# -*- coding: utf-8 -*-
"""
Bayesian Online Changepoint Detection (Adams & MacKay) for a 1-D stream.
We use a Normal-Gamma prior (unknown mean/variance) → Student-t predictive.

Input:  e[0..T-1]  (recommend: z-scored or robust-z-scored per episode)
Knob:   hazard H in (0,1)   ≈ 1 / expected_segment_length_in_frames
Limit:  rmax caps the run-length to keep O(T * rmax) time/memory.

Returns: cp_prob[t] = P(changepoint at t) = posterior run-length == 0
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from scipy.special import gammaln

# ---------------- utilities ----------------

def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Median/MAD z-score, robust to outliers."""
    m = np.median(x)
    mad = np.median(np.abs(x - m)) + eps
    return (x - m) / (1.4826 * mad)

def student_t_pdf(x: np.ndarray, mu: np.ndarray, lam: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Vectorized Student-t predictive from Normal-Gamma posterior.
    lam = kappa  (precision scaling of the mean)
    df  = 2*alpha
    scale^2 = beta * (lam + 1) / (alpha * lam)
    """
    # broadcast x to match parameter arrays
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = np.full_like(mu, float(x))
    else:
        x = np.broadcast_to(x, mu.shape)

    df = 2.0 * alpha
    scale2 = (beta * (lam + 1.0)) / (alpha * lam)  # variance-like term

    # log-pdf for numerical stability
    # log c = lgamma((df+1)/2) - lgamma(df/2) - 0.5*(log(df) + log(pi) + log(scale2))
    logc = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * (np.log(df) + np.log(np.pi) + np.log(scale2))
    t2 = ((x - mu) ** 2) / (df * scale2)
    logpdf = logc - 0.5 * (df + 1.0) * np.log1p(t2)

    pdf = np.exp(logpdf)
    # safety clamp (no zeros / nans)
    pdf = np.clip(pdf, 1e-300, 1.0)
    return pdf


@dataclass
class BOCPDConfig:
    hazard: float = 0.005     # ~ 1 / expected segment length (in frames)
    rmax: int = 300           # max tracked run length
    use_robust_z: bool = True # robust-z the input first
    # Normal-Gamma prior hyperparams (weak, symmetric)
    mu0: float = 0.0
    kappa0: float = 1e-3
    alpha0: float = 1.0
    beta0: float = 1.0

def bocpd(e: np.ndarray, cfg: BOCPDConfig) -> np.ndarray:
    """
    Core BOCPD recursion with run-length truncation.
    e: 1-D array (float32/float64), recommendation: pre-zscored per episode.
    Returns: cp_prob[t] in [0,1]
    """
    x = e.astype(np.float64)
    if cfg.use_robust_z:
        x = robust_zscore(x)

    T = x.shape[0]
    R = np.zeros((T, cfg.rmax + 1), dtype=np.float64)     # run-length posterior
    R[0, 0] = 1.0

    # Sufficient stats for each possible run length r at time t:
    mu = np.full(cfg.rmax + 1, cfg.mu0, dtype=np.float64)
    kappa = np.full(cfg.rmax + 1, cfg.kappa0, dtype=np.float64)
    alpha = np.full(cfg.rmax + 1, cfg.alpha0, dtype=np.float64)
    beta  = np.full(cfg.rmax + 1, cfg.beta0,  dtype=np.float64)

    cp_prob = np.zeros(T, dtype=np.float32)
    H = float(cfg.hazard)
    one_minus_H = 1.0 - H

    # time step 0: predictive irrelevant; record cp_prob[0]=1 if you want a start-cut, else 0
    cp_prob[0] = 0.0

    for t in range(1, T):
        xt = x[t]

        # 1) Predictive for all current run-lengths (0..rmax-1)
        #    For r = 0..rmax-1 at time t-1, we have posterior params (mu,kappa,alpha,beta)
        pred = student_t_pdf(xt, mu, kappa, alpha, beta)  # shape [rmax+1]

        # 2) Growth: run-length r -> r+1 (shifted by one), weighted by (1-H)
        growth = R[t-1, :cfg.rmax] * pred[:cfg.rmax] * one_minus_H
        R[t, 1:cfg.rmax+1] = growth

        # 3) Changepoint at t: run-length resets to 0, sum over all previous r
        R[t, 0] = np.sum(R[t-1, :] * pred * H)

        # 4) Normalize
        Z = np.sum(R[t, :])
        if Z <= 0 or not np.isfinite(Z):
            # numerical safety fallback: reset to prior at this frame
            R[t, :] = 0.0
            R[t, 0] = 1.0
            Z = 1.0
        R[t, :] /= Z

        cp_prob[t] = float(R[t, 0])

        # 5) Update posterior params for next step:
        #    We need params for run-length r+1 at time t (based on r at t-1 updated with x_t)
        #    Vectorized update for all r=0..rmax-1:
        mu_new    = (kappa * mu + xt) / (kappa + 1.0)
        kappa_new = kappa + 1.0
        alpha_new = alpha + 0.5
        beta_new  = beta + 0.5 * ( (kappa * (xt - mu) ** 2) / (kappa + 1.0) )

        # shift them into positions 1..rmax for next iteration
        mu[1:]    = mu_new[:-1]
        kappa[1:] = kappa_new[:-1]
        alpha[1:] = alpha_new[:-1]
        beta[1:]  = beta_new[:-1]

        # position 0 (new run starting at t) resets to the prior
        mu[0]    = cfg.mu0
        kappa[0] = cfg.kappa0
        alpha[0] = cfg.alpha0
        beta[0]  = cfg.beta0

    # optional smoothing of cp_prob (light)
    # cp_prob = np.clip(cp_prob, 0, 1)
    return cp_prob
