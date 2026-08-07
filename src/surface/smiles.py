"""Per-expiry implied-volatility smiles.

For one expiry we take the forward from `forwards.infer_forward`, invert each option's
market mid to a Black-Scholes implied vol with our own solver (src.bs), and express the
result in two coordinate systems the desk uses:

  * log-moneyness  k = ln(K / F)     — SVI's native x-axis
  * Black-Scholes delta               — the trader's smile axis (25-delta RR/BF live here)

We deliberately price each option on its OWN side (calls with call IV, puts with put IV)
using the parity-consistent forward and r = -ln(df)/tau, so a call and put at the same
strike give the same IV up to bid-ask noise. Points where the solver returns None
(sub-intrinsic mid, vega collapse deep OTM) are dropped and counted — honest-unknown,
never a fabricated vol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.bs import BS
from src.schema import OptionQuote
from src.surface.forwards import ForwardFit


@dataclass(frozen=True, slots=True)
class SmilePoint:
    strike: float
    log_moneyness: float       # ln(K / F)
    iv: float                  # market-implied vol (decimal) from our solver
    total_variance: float      # w = iv^2 * tau  (SVI's native quantity)
    delta: float               # BS delta of this option (signed: calls>0, puts<0)
    option_type: str           # "C" or "P"
    used_mid: bool             # True if a real two-sided mid was used (else mark fallback)


@dataclass(frozen=True, slots=True)
class Smile:
    underlying: str
    expiry_ts: float
    tau: float
    forward: float
    rate: float
    points: tuple[SmilePoint, ...]
    n_dropped: int             # quotes whose IV could not be solved (reported, not hidden)

    @property
    def log_moneyness(self) -> np.ndarray:
        return np.array([p.log_moneyness for p in self.points])

    @property
    def total_variance(self) -> np.ndarray:
        return np.array([p.total_variance for p in self.points])

    @property
    def ivs(self) -> np.ndarray:
        return np.array([p.iv for p in self.points])


def _usd_mid_or_mark(q: OptionQuote) -> tuple[float | None, bool]:
    """Return (USD premium, used_mid). Prefer a real mid; fall back to mark."""
    if q.mid_coin is not None and q.mid_coin > 0.0:
        return q.mid_coin * q.index_price, True
    if q.mark_price_coin > 0.0:
        return q.mark_price_coin * q.index_price, False
    return None, False


def build_smile(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
    fit: ForwardFit,
    *,
    otm_only: bool = True,
) -> Smile:
    """Build one expiry's smile from its quotes and the parity-inferred forward.

    `otm_only` (the default) keeps the liquid out-of-the-money wing of each side (calls for
    K>=F, puts for K<F) — the standard smile construction, avoiding the illiquid deep-ITM
    options and the double-counting of ITM/OTM at the same strike. With `otm_only=True` the
    ITM points are skipped entirely; pass `otm_only=False` to retain both sides at every
    strike (both are individually solvable).
    """
    F = fit.forward
    tau = fit.tau
    r = fit.rate
    spot = F * math.exp(-r * tau)  # spot consistent with F = spot*exp(r*tau), carry 0

    points: list[SmilePoint] = []
    dropped = 0
    for q in quotes:
        if otm_only:
            if q.option_type == "C" and q.strike < F:
                continue
            if q.option_type == "P" and q.strike >= F:
                continue
        usd, used_mid = _usd_mid_or_mark(q)
        if usd is None:
            dropped += 1
            continue
        iv = BS.implied_vol(
            price=usd, spot=spot, strike=q.strike, tau=tau, rate=r,
            option_type=q.option_type, carry=0.0,
        )
        if iv is None or iv <= 0.0:
            dropped += 1
            continue
        k = math.log(q.strike / F)
        g = BS.greeks(spot=spot, strike=q.strike, tau=tau, rate=r, sigma=iv,
                      option_type=q.option_type, carry=0.0)
        points.append(SmilePoint(
            strike=q.strike, log_moneyness=k, iv=iv, total_variance=iv * iv * tau,
            delta=g.delta, option_type=q.option_type, used_mid=used_mid,
        ))

    points.sort(key=lambda p: p.log_moneyness)
    return Smile(
        underlying=fit.underlying, expiry_ts=fit.expiry_ts, tau=tau,
        forward=F, rate=r, points=tuple(points), n_dropped=dropped,
    )
