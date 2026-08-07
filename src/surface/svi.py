"""SVI smile calibration under no-arbitrage constraints (ORCH-owned, W2).

Raw SVI (Gatheral 2004) parameterizes total implied variance as a function of
log-moneyness k = ln(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

with parameters (a, b, rho, m, sigma):
  a     : overall variance level (vertical shift)
  b     : angle between the asymptotes (b >= 0)
  rho   : asymmetry / skew (-1 < rho < 1)
  m     : horizontal shift of the smile minimum
  sigma : smile curvature at the minimum (sigma > 0)

No-arbitrage discipline (documented in DESIGN.md):
  * Domain constraints keep the slice from generating negative variance and cap the wing
    slopes by the Lee moment bound:  b*(1+|rho|) <= 4/(1+|rho|) is enforced via the
    upper bound  b*(1+|rho|)*tau <= 4  (calendar-scaled). a + b*sigma*sqrt(1-rho^2) >= 0
    keeps w(k) >= 0 everywhere.
  * Butterfly (static) arbitrage is checked AFTER the fit via Gatheral's g(k) function
    (see src/noarb); a slice with g(k) < 0 is flagged, never silently smoothed.

Fit: least squares on total variance in k-space (weighted by liquidity), with a small
multi-start over m/sigma to avoid the well-known local minima. Deterministic (fixed
starts, no RNG). Reports RMSE in total-variance and in vol-point space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "sigma": self.sigma}


@dataclass(frozen=True, slots=True)
class SVIFit:
    params: SVIParams
    rmse_w: float          # RMSE in total-variance space
    rmse_vol: float        # RMSE in vol-point space (decimal) at the fit points
    n_points: int
    tau: float
    converged: bool


def svi_total_variance(k, params: SVIParams):
    """Raw-SVI total variance w(k). Accepts scalar or array k."""
    k = np.asarray(k, dtype=float)
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_iv(k, params: SVIParams, tau: float):
    """Implied vol (decimal) from the SVI slice: sqrt(w(k)/tau), clipped at 0."""
    w = np.maximum(svi_total_variance(k, params), 0.0)
    return np.sqrt(w / tau)


def _initial_guesses(k: np.ndarray, w: np.ndarray) -> list[SVIParams]:
    """A few deterministic starting points spanning plausible skew/curvature."""
    w_min = float(np.min(w))
    k_at_min = float(k[np.argmin(w)])
    spread = float(np.ptp(k)) or 0.1
    guesses = []
    for rho0 in (-0.5, 0.0, -0.2):
        for sig0 in (spread * 0.5, spread):
            guesses.append(SVIParams(
                a=max(w_min * 0.5, 1e-6),
                b=max((float(np.ptp(w)) / spread) if spread else 0.1, 1e-3),
                rho=rho0, m=k_at_min, sigma=max(sig0, 1e-3),
            ))
    return guesses


def calibrate_svi(
    log_moneyness,
    total_variance,
    tau: float,
    *,
    weights=None,
) -> SVIFit:
    """Calibrate raw SVI to (k, w) points under no-arbitrage domain constraints.

    Deterministic multi-start bounded least squares. `weights` (e.g. inverse spread)
    scale each residual. Butterfly g(k) is validated separately by src/noarb.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if weights is None:
        wt = np.ones_like(k)
    else:
        wt = np.asarray(weights, dtype=float)
    n = len(k)
    if n < 5:
        raise ValueError(f"SVI needs >= 5 points for a stable 5-param fit, got {n}")

    sw = np.sqrt(wt)

    def objective(theta: np.ndarray) -> float:
        a, b, rho, m, sigma = theta
        params = SVIParams(a, b, rho, m, sigma)
        model = svi_total_variance(k, params)
        return float(np.sum((sw * (model - w)) ** 2))

    # Bounds enforce the domain: b>=0, -1<rho<1, sigma>0, a can be slightly negative but
    # the a + b*sigma*sqrt(1-rho^2) >= 0 constraint (below) keeps w(k)>=0 globally.
    k_lo, k_hi = float(np.min(k)), float(np.max(k))
    bounds = [
        (-0.5, max(1.0, 2 * float(np.max(w)))),    # a
        (0.0, 4.0 / max(tau, 1e-6)),                # b (Lee-bound-scaled ceiling)
        (-0.999, 0.999),                            # rho
        (k_lo - 1.0, k_hi + 1.0),                   # m
        (1e-4, 5.0),                                # sigma
    ]

    def constraint_wpos(theta: np.ndarray) -> float:
        a, b, rho, _m, sigma = theta
        # Global minimum of w is a + b*sigma*sqrt(1-rho^2); require >= 0.
        return a + b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0))

    def constraint_lee(theta: np.ndarray) -> float:
        # Wing slopes bounded: b*(1+|rho|)*tau <= 4 (no-arb asymptotic slope <= 2 each side).
        _a, b, rho, _m, _sigma = theta
        return 4.0 - b * (1.0 + abs(rho)) * tau

    cons = [
        {"type": "ineq", "fun": constraint_wpos},
        {"type": "ineq", "fun": constraint_lee},
    ]

    best: tuple[float, np.ndarray] | None = None
    for g in _initial_guesses(k, w):
        x0 = np.array([g.a, g.b, g.rho, g.m, g.sigma])
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x)

    assert best is not None
    theta = best[1]
    params = SVIParams(*theta)
    model = svi_total_variance(k, params)
    rmse_w = float(np.sqrt(np.mean((model - w) ** 2)))
    model_vol = np.sqrt(np.maximum(model, 0.0) / tau)
    mkt_vol = np.sqrt(np.maximum(w, 0.0) / tau)
    rmse_vol = float(np.sqrt(np.mean((model_vol - mkt_vol) ** 2)))
    converged = constraint_wpos(theta) >= -1e-9 and constraint_lee(theta) >= -1e-9

    return SVIFit(params=params, rmse_w=rmse_w, rmse_vol=rmse_vol,
                  n_points=n, tau=tau, converged=converged)
