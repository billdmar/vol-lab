"""Parity-inferred forwards per expiry (ORCH-owned, W2).

The forward is not assumed — it is *inferred* from put-call parity on liquid strikes,
which is exactly the interview talking point the mission wants demonstrated.

Method (documented in docs/DESIGN.md):
  Put-call parity in USD terms:  C(K) - P(K) = df * (F - K),
  so a linear regression of (C - P) on K has slope = -df and intercept = df * F.
  Hence df = -slope and F = intercept / df.

  We regress on the C/P pairs whose *both* legs have a two-sided market (a real mid),
  falling back to the exchange mark only when no mid exists, and weight each pair by an
  inverse-spread liquidity weight so wide/stale strikes don't drag the fit. The number
  of pairs used and any dropped are reported (never silently discarded).

Validation: the inferred F is cross-checked against Deribit's own per-expiry
`underlying_price` (its published forward); the differences are reported in W2/G2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schema import OptionQuote

# ACT/365 day count (frozen convention). Deribit options settle 08:00 UTC (in expiry_ts).
SECONDS_PER_YEAR = 365.0 * 86400.0


@dataclass(frozen=True, slots=True)
class ForwardFit:
    """Result of inferring the forward for one expiry from put-call parity."""

    underlying: str
    expiry_ts: float
    tau: float                 # ACT/365 year fraction
    forward: float             # inferred F (USD)
    discount_factor: float     # inferred df = -slope
    n_pairs: int               # C/P strike pairs used in the regression
    n_pairs_available: int     # pairs that existed before liquidity screening
    rate: float                # implied continuous rate: -ln(df)/tau (0 if df~1)
    resid_rms: float           # RMS of C-P vs df*(F-K) over used pairs (USD)
    deribit_forward: float | None  # exchange underlying_price for cross-check, if present


def year_fraction(expiry_ts: float, ref_ts: float) -> float:
    """ACT/365 year fraction from a reference (snapshot) time to expiry."""
    return max((expiry_ts - ref_ts) / SECONDS_PER_YEAR, 0.0)


def _usd_price(q: OptionQuote) -> float | None:
    """Best available USD premium: coin mid x index if a mid exists, else mark x index."""
    coin = q.mid_coin if q.mid_coin is not None else q.mark_price_coin
    if coin is None or coin <= 0.0:
        return None
    return coin * q.index_price


def infer_forward(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
    *,
    ref_ts: float,
    min_pairs: int = 3,
) -> ForwardFit | None:
    """Infer the forward for a single expiry's quotes via a parity regression.

    All `quotes` must share one underlying and one expiry. Returns None if fewer than
    `min_pairs` usable C/P pairs exist (honest-unknown — we do not fabricate a forward
    from one strike).
    """
    if not quotes:
        return None
    underlying = quotes[0].underlying
    expiry_ts = quotes[0].expiry_ts
    tau = year_fraction(expiry_ts, ref_ts)
    if tau <= 0.0:
        return None

    by_strike: dict[float, dict[str, OptionQuote]] = {}
    for q in quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q

    strikes: list[float] = []
    diffs: list[float] = []      # C - P in USD
    weights: list[float] = []
    n_available = 0
    for k, cp in sorted(by_strike.items()):
        if "C" not in cp or "P" not in cp:
            continue
        n_available += 1
        cu = _usd_price(cp["C"])
        pu = _usd_price(cp["P"])
        if cu is None or pu is None:
            continue
        # Liquidity weight: inverse of the summed relative spreads (tight markets weigh more).
        rs_c = cp["C"].rel_spread
        rs_p = cp["P"].rel_spread
        rs = (rs_c if rs_c is not None else 1.0) + (rs_p if rs_p is not None else 1.0)
        weights.append(1.0 / (1.0 + rs))
        strikes.append(k)
        diffs.append(cu - pu)

    if len(strikes) < min_pairs:
        return None

    x = np.asarray(strikes, dtype=float)
    y = np.asarray(diffs, dtype=float)
    w = np.asarray(weights, dtype=float)

    # Weighted least squares for slope/intercept of  y = slope*K + intercept.
    sw = np.sqrt(w)
    a_mat = np.vstack([x * sw, sw]).T
    b_vec = y * sw
    (slope, intercept), *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)

    df = -slope
    if df <= 0.0:
        return None  # nonsensical (would imply negative discount) — report as no-fit
    forward = intercept / df

    resid = y - (slope * x + intercept)
    resid_rms = float(np.sqrt(np.mean(resid**2)))
    # Continuous rate implied by the discount factor (r=0 when df~1, the crypto default).
    rate = float(-np.log(df) / tau) if df < 1.0 else 0.0

    deribit_fwd = quotes[0].underlying_price if quotes[0].underlying_price > 0 else None

    return ForwardFit(
        underlying=underlying,
        expiry_ts=expiry_ts,
        tau=tau,
        forward=float(forward),
        discount_factor=float(df),
        n_pairs=len(strikes),
        n_pairs_available=n_available,
        rate=rate,
        resid_rms=resid_rms,
        deribit_forward=deribit_fwd,
    )


def infer_forwards_all_expiries(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
    *,
    ref_ts: float,
    min_pairs: int = 3,
) -> dict[float, ForwardFit]:
    """Infer forwards for every expiry present in `quotes` (one underlying).

    Returns {expiry_ts: ForwardFit} for expiries with enough liquid pairs. Expiries that
    fail the min_pairs screen are omitted (and can be counted by the caller for reporting).
    """
    by_expiry: dict[float, list[OptionQuote]] = {}
    for q in quotes:
        by_expiry.setdefault(q.expiry_ts, []).append(q)
    out: dict[float, ForwardFit] = {}
    for e, qs in by_expiry.items():
        fit = infer_forward(qs, ref_ts=ref_ts, min_pairs=min_pairs)
        if fit is not None:
            out[e] = fit
    return out
