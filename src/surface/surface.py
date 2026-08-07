"""Full-surface construction: forwards -> smiles -> SVI per expiry (ORCH-owned, W2).

Ties the pieces together for one underlying in one snapshot and exposes the descriptive
quantities the research note needs: per-expiry SVI params + RMSE, the ATM vol term
structure, and the 25-delta risk-reversal and butterfly (the desk's smile summary).

25-delta conventions (documented in DESIGN.md):
  * RR_25 = IV(25d call) - IV(25d put)      (skew: negative => puts richer)
  * BF_25 = 0.5*(IV(25d call)+IV(25d put)) - IV(ATM)   (smile convexity / wings)
  IVs are read off the *calibrated* SVI slice at the log-moneyness where BS delta = +/-0.25,
  found by a bracketed root solve, so RR/BF are smooth surface quantities, not noisy quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.schema import OptionQuote
from src.surface.forwards import ForwardFit, infer_forwards_all_expiries
from src.surface.smiles import Smile, build_smile
from src.surface.svi import SVIFit, SVIParams, calibrate_svi, svi_iv


@dataclass(frozen=True, slots=True)
class ExpirySlice:
    expiry_ts: float
    tau: float
    forward: float
    fit: ForwardFit
    smile: Smile
    svi: SVIFit
    atm_vol: float             # SVI vol at k=0 (forward-ATM)
    rr_25: float | None        # 25-delta risk reversal (vol points, decimal)
    bf_25: float | None        # 25-delta butterfly (vol points, decimal)


@dataclass(frozen=True, slots=True)
class Surface:
    underlying: str
    snapshot_ts: float
    slices: tuple[ExpirySlice, ...]

    @property
    def term_structure(self) -> list[tuple[float, float]]:
        """[(tau, atm_vol)] sorted by tau — the ATM term structure."""
        return sorted((s.tau, s.atm_vol) for s in self.slices)


def _k_at_delta(target_delta: float, params: SVIParams, tau: float, forward: float,
                rate: float, option_type: str) -> float | None:
    """Find log-moneyness k where the SVI-implied BS delta equals target_delta.

    Solves BS_delta(k; sigma=SVI(k)) = target on the OTM side. Returns None if no bracket.
    """
    spot = forward * np.exp(-rate * tau)

    def delta_at_k(k: float) -> float:
        sig = float(svi_iv(k, params, tau))
        if sig <= 0.0:
            return np.nan
        strike = forward * np.exp(k)
        d1 = (np.log(spot / strike) + (rate + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
        if option_type == "C":
            return float(np.exp(-0.0 * tau) * norm.cdf(d1))  # carry 0
        return float(np.exp(-0.0 * tau) * (norm.cdf(d1) - 1.0))

    # Call delta in (0,1) decreasing in k; put delta in (-1,0) decreasing in |k|.
    lo, hi = -3.0, 3.0
    try:
        f = lambda k: delta_at_k(k) - target_delta  # noqa: E731
        flo, fhi = f(lo), f(hi)
        if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
            return None
        return float(brentq(f, lo, hi, maxiter=200, xtol=1e-8))
    except (ValueError, RuntimeError):
        return None


def _rr_bf_25(params: SVIParams, tau: float, forward: float, rate: float,
              atm_vol: float) -> tuple[float | None, float | None]:
    """25-delta risk-reversal and butterfly read off the calibrated SVI slice."""
    k_call = _k_at_delta(0.25, params, tau, forward, rate, "C")
    k_put = _k_at_delta(-0.25, params, tau, forward, rate, "P")
    if k_call is None or k_put is None:
        return None, None
    iv_c = float(svi_iv(k_call, params, tau))
    iv_p = float(svi_iv(k_put, params, tau))
    rr = iv_c - iv_p
    bf = 0.5 * (iv_c + iv_p) - atm_vol
    return rr, bf


def build_surface(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
    *,
    ref_ts: float,
    min_smile_points: int = 5,
) -> Surface:
    """Build the calibrated SVI surface for one underlying from one snapshot's quotes.

    Expiries without a parity forward or with too few smile points are skipped (counted by
    the caller via len(surface.slices) vs available expiries). Deterministic end to end.
    """
    if not quotes:
        raise ValueError("no quotes provided")
    underlying = quotes[0].underlying
    snapshot_ts = quotes[0].snapshot_ts

    forwards = infer_forwards_all_expiries(quotes, ref_ts=ref_ts)
    by_expiry: dict[float, list[OptionQuote]] = {}
    for q in quotes:
        by_expiry.setdefault(q.expiry_ts, []).append(q)

    slices: list[ExpirySlice] = []
    for e, fit in sorted(forwards.items()):
        smile = build_smile(by_expiry[e], fit)
        if len(smile.points) < min_smile_points:
            continue
        # Liquidity weights: real mids weigh more than mark-fallbacks.
        weights = np.array([1.0 if p.used_mid else 0.3 for p in smile.points])
        svi = calibrate_svi(smile.log_moneyness, smile.total_variance, fit.tau,
                            weights=weights)
        atm_vol = float(svi_iv(0.0, svi.params, fit.tau))
        rr, bf = _rr_bf_25(svi.params, fit.tau, fit.forward, fit.rate, atm_vol)
        slices.append(ExpirySlice(
            expiry_ts=e, tau=fit.tau, forward=fit.forward, fit=fit, smile=smile,
            svi=svi, atm_vol=atm_vol, rr_25=rr, bf_25=bf,
        ))

    return Surface(underlying=underlying, snapshot_ts=snapshot_ts, slices=tuple(slices))
