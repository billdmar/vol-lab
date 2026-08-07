"""Black-Scholes-Merton closed form: pricer, Greeks, and implied-vol solver.

This is the reference engine for vol-lab. Every other engine (CRR lattice, Monte
Carlo, LSMC) is cross-verified against these numbers, and these closed-form Greeks
are one leg of the G1 three-way Greeks reconciliation, so the formulas are kept
textbook-exact under the frozen numerical conventions (see src/interfaces.py):

  * Forward  F  = spot * exp((rate - carry) * tau)      (carry q = cost-of-carry / div yield)
  * Discount df = exp(-rate * tau)
  * Greeks units: delta = dP/dSpot, gamma = d2P/dSpot2, vega = dP/dSigma (per 1.00 vol),
    theta = dP/dt (per YEAR, negative for long options), rho = dP/dRate (per 1.00 rate).
  * option_type is "C" or "P"; IV is a decimal (0.65 == 65%).

Normal CDF uses math.erfc (accurate deep in the tails, where 0.5*(1+erf) loses digits),
which matters for the deep-OTM regime that stresses the IV solver.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from src.interfaces import Greeks, PriceResult
from src.schema import OptionType

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erfc (tail-accurate)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


class BlackScholes:
    """Closed-form Black-Scholes-Merton engine.

    Implements the Pricer, GreeksEngine and IVSolver protocols. Stateless: every
    call is a pure function of its keyword arguments, so instances are trivially
    reusable and thread-safe. A single instance is exported as `BS` below.
    """

    # --------------------------------------------------------------- pricing
    def price(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> PriceResult:
        """Black-Scholes-Merton price on the parity forward.

        Degenerate limits (handled explicitly, no NaN/inf):
          * tau <= 0    -> discounted intrinsic on the spot (df == 1 at expiry).
          * sigma <= 0  -> deterministic forward payoff, df * max(F - K, 0) (call).
        """
        _validate(spot=spot, strike=strike, tau=tau, sigma=sigma, option_type=option_type)
        df = math.exp(-rate * tau) if tau > 0.0 else 1.0
        fwd = spot * math.exp((rate - carry) * tau)

        # Deterministic limits: no diffusion, value is the discounted forward payoff.
        if tau <= 0.0 or sigma <= 0.0 or sigma * math.sqrt(tau) < 1e-16:
            if option_type == "C":
                px = df * max(fwd - strike, 0.0)
            else:
                px = df * max(strike - fwd, 0.0)
            return PriceResult(price=px)

        d1, d2 = _d1_d2(fwd, strike, sigma, tau)
        if option_type == "C":
            px = df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
        else:
            px = df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))
        return PriceResult(price=px)

    # ---------------------------------------------------------------- Greeks
    def greeks(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> Greeks:
        """Closed-form first/second-order sensitivities in the frozen units.

        Degenerate limits return finite values (gamma = vega = 0 with no diffusion);
        they are not the focus of the G1 reconciliation, which uses regular inputs.
        """
        _validate(spot=spot, strike=strike, tau=tau, sigma=sigma, option_type=option_type)
        eqt = math.exp(-carry * tau) if tau > 0.0 else 1.0  # e^{-q*tau}
        df = math.exp(-rate * tau) if tau > 0.0 else 1.0
        fwd = spot * math.exp((rate - carry) * tau)

        # Deterministic limit: step-function delta, no gamma/vega. rho/theta from the
        # surviving discount terms (N(d1)=N(d2)-> 1 when in-the-money on the forward).
        if tau <= 0.0 or sigma <= 0.0 or sigma * math.sqrt(tau) < 1e-16:
            itm = fwd > strike if option_type == "C" else fwd < strike
            if option_type == "C":
                delta = eqt if itm else 0.0
                theta = (carry * spot * eqt - rate * strike * df) if itm else 0.0
                rho = strike * tau * df if itm else 0.0
            else:
                delta = -eqt if itm else 0.0
                theta = (-carry * spot * eqt + rate * strike * df) if itm else 0.0
                rho = -strike * tau * df if itm else 0.0
            return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=theta, rho=rho)

        sqrt_t = math.sqrt(tau)
        d1, d2 = _d1_d2(fwd, strike, sigma, tau)
        pdf_d1 = _norm_pdf(d1)

        # gamma and vega are option-type independent.
        gamma = eqt * pdf_d1 / (spot * sigma * sqrt_t)
        vega = spot * eqt * pdf_d1 * sqrt_t
        # Shared diffusion decay term of theta (per year): -S e^{-q t} n(d1) sigma / (2 sqrt(t)).
        decay = -(spot * eqt * pdf_d1 * sigma) / (2.0 * sqrt_t)

        if option_type == "C":
            delta = eqt * _norm_cdf(d1)
            theta = decay + carry * spot * eqt * _norm_cdf(d1) - rate * strike * df * _norm_cdf(d2)
            rho = strike * tau * df * _norm_cdf(d2)
        else:
            delta = -eqt * _norm_cdf(-d1)
            theta = (
                decay
                - carry * spot * eqt * _norm_cdf(-d1)
                + rate * strike * df * _norm_cdf(-d2)
            )
            rho = -strike * tau * df * _norm_cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    # ----------------------------------------------------------- IV solver
    def implied_vol(
        self,
        *,
        price: float,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> float | None:
        """Invert a European price to Black-Scholes implied vol (decimal), or None.

        Method: no-arbitrage bound check -> bracket sigma so the price is straddled ->
        safeguarded Newton (rtsafe) with a VEGA FLOOR that falls back to bisection when
        vega is tiny -> Brent as a final bracketed fallback. Converges the *price* to
        ~1e-8, which bounds the vol round-trip error under 1e-6 in the well-behaved
        regime (see TOL["iv_roundtrip_abs"]).

        Returns None (honest-unknown, never a fabricated vol) when:
          * tau <= 0                    -- an expired option carries no vol information.
          * price < discounted intrinsic -- sub-intrinsic, no arbitrage-free IV exists.
          * price >= the no-arb ceiling  -- call >= spot*e^{-q*tau} / put >= strike*df.
          * price is at/below intrinsic within float resolution -- deep-OTM / near-expiry
            VEGA COLLAPSE: the price (often an underflow to 0.0) pins sigma at ~0 and
            carries no recoverable vol. We return None rather than a meaningless ~0 vol.
        """
        _validate(spot=spot, strike=strike, tau=tau, sigma=1.0, option_type=option_type)
        if tau <= 0.0:
            return None

        df = math.exp(-rate * tau)
        eqt = math.exp(-carry * tau)
        fwd = spot * math.exp((rate - carry) * tau)

        # No-arbitrage price envelope. Intrinsic (discounted) is the sigma->0 floor;
        # the ceiling is the sigma->inf limit (call -> spot*e^{-q t}, put -> K*df).
        if option_type == "C":
            intrinsic = df * max(fwd - strike, 0.0)
            ceiling = spot * eqt
        else:
            intrinsic = df * max(strike - fwd, 0.0)
            ceiling = strike * df

        price_floor = 1e-12  # float-resolution band around the intrinsic / ceiling
        if price < intrinsic - price_floor:
            return None  # sub-intrinsic: arbitrage, no IV
        if price >= ceiling - price_floor:
            return None  # at/above the no-arb ceiling: sigma -> inf
        if price <= intrinsic + price_floor:
            return None  # vega collapse: price carries no recoverable vol

        target = price

        def _px_vega(sig: float) -> tuple[float, float]:
            d1, d2 = _d1_d2(fwd, strike, sig, tau)
            if option_type == "C":
                px = df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
            else:
                px = df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))
            vega = spot * eqt * _norm_pdf(d1) * math.sqrt(tau)
            return px, vega

        # Bracket: f(sig) = px(sig) - target is monotone increasing (vega > 0).
        # f(lo) ~ intrinsic - target < 0 by the no-arb checks above, so grow hi until it straddles.
        lo, hi = 1e-9, 1.0
        f_hi = _px_vega(hi)[0] - target
        grow = 0
        while f_hi < 0.0 and hi < 50.0:
            hi *= 2.0
            f_hi = _px_vega(hi)[0] - target
            grow += 1
            if grow > 64:  # pragma: no cover - guarded by the 50.0 cap
                break
        if f_hi < 0.0:  # pragma: no cover - defensive: ceiling check makes hi straddle by 50.0
            # Numerical vega collapse near sigma_max; the no-arb ceiling check precludes this.
            return None

        # Safeguarded Newton (rtsafe): keep [lo, hi] straddling the root; Newton when
        # vega is healthy, bisection when vega < floor or a step escapes the bracket.
        vega_floor = 1e-12
        price_tol = 1e-8
        sig = 0.5 * (lo + hi)
        for _ in range(100):
            px, vega = _px_vega(sig)
            diff = px - target
            if diff > 0.0:
                hi = sig
            else:
                lo = sig
            if abs(diff) < price_tol:
                return sig
            if vega < vega_floor:
                sig = 0.5 * (lo + hi)
                continue
            step = diff / vega
            nxt = sig - step
            if not (lo < nxt < hi):
                nxt = 0.5 * (lo + hi)  # Newton escaped the bracket -> bisect
            sig = nxt

        # Final fallback: Brent on the established bracket, then verify the price. The
        # embedded bisection guarantees convergence within the 100-iteration budget, so
        # this tail is defensive only (never exercised in practice, hence no-cover).
        try:  # pragma: no cover
            root = brentq(lambda s: _px_vega(s)[0] - target, lo, hi, xtol=1e-12, maxiter=200)
        except (ValueError, RuntimeError):  # pragma: no cover
            return None
        return root if abs(_px_vega(root)[0] - target) < 1e-6 else None  # pragma: no cover


def _d1_d2(fwd: float, strike: float, sigma: float, tau: float) -> tuple[float, float]:
    """Black-Scholes d1, d2 in forward form (F, K)."""
    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(fwd / strike) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def _validate(
    *, spot: float, strike: float, tau: float, sigma: float, option_type: OptionType
) -> None:
    """Guard against inputs that would silently produce NaN/inf (fail loud instead)."""
    if option_type not in ("C", "P"):
        raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")
    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be positive, got {strike}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")


# Module-level singleton for convenient reuse across engines/tests.
BS = BlackScholes()
