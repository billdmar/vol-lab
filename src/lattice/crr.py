"""Cox-Ross-Rubinstein binomial tree pricer (European + American).

Implements the frozen `Pricer` interface (src/interfaces.py) so its prices are
apples-to-apples with the closed form, Monte Carlo, and LSMC engines. The tree is
built on the SAME frozen numerical conventions as every other engine:

    forward F  = spot * exp((rate - carry) * tau)
    discount   = exp(-rate * tau)
    per step   dt = tau / N
    up move    u  = exp(sigma * sqrt(dt))
    down move  d  = 1 / u
    risk-neutral  p = (exp((rate - carry) * dt) - d) / (u - d)

The one-step growth factor is exp((rate - carry) * dt) rather than exp(rate*dt):
this is the cost-of-carry drift so the tree's forward equals F exactly, which is
what makes CRR agree with Black-Scholes (and keeps European put-call parity exact
on the tree). See docs/DESIGN.md for the carry convention.

American options take max(continuation, intrinsic) at every node — the standard
early-exercise test on a recombining lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.common import intrinsic as _intrinsic
from src.interfaces import Pricer, PriceResult
from src.schema import OptionType


@dataclass(frozen=True, slots=True)
class CRRBinomial:
    """CRR binomial-tree pricer for European and American vanilla options.

    Parameters
    ----------
    steps:
        Number of time steps N in the tree (default 2000). More steps -> tighter
        convergence to the continuous-time (Black-Scholes) price at O(1/N).
    american:
        If True, allow early exercise (max of continuation vs intrinsic at each
        node). If False (default), price the European option via pure backward
        induction.
    """

    steps: int = 2000
    american: bool = False

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
        """Price a vanilla option on the CRR tree. Deterministic -> no stderr/ci95."""
        value = _crr_price(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=rate,
            sigma=sigma,
            option_type=option_type,
            carry=carry,
            steps=self.steps,
            american=self.american,
        )
        return PriceResult(price=value)


def _crr_price(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType,
    carry: float,
    steps: int,
    american: bool,
) -> float:
    """Core CRR backward induction. Returns the option value at the root node.

    Handles the tau == 0 degenerate case (return intrinsic) and validates that the
    risk-neutral probability is a genuine probability in [0, 1]; a p outside that
    band means the up/down moves cannot span the risk-neutral drift (arbitrageable
    tree) and we raise rather than return a plausible-but-wrong number.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0 for the CRR tree, got {sigma}")
    if tau < 0.0:
        raise ValueError(f"tau must be >= 0, got {tau}")

    # At/after expiry the option is worth its intrinsic value with certainty.
    if tau == 0.0:
        return float(_intrinsic(np.array([spot]), strike, option_type)[0])

    dt = tau / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((rate - carry) * dt)   # cost-of-carry drift per step
    p = (growth - d) / (u - d)

    # A valid recombining tree needs 0 <= p <= 1. Allow a hair of float slack; a
    # real excursion means dt is too coarse for these (rate, carry, sigma) -> the
    # honest answer is to fail loudly, not to clip into a fake probability.
    if not (-1e-12 <= p <= 1.0 + 1e-12):
        raise ValueError(
            f"risk-neutral p={p:.6g} outside [0,1] (dt={dt:.3e}, u={u:.6g}); "
            "tree is arbitrageable — increase steps or check inputs."
        )
    p = min(1.0, max(0.0, p))
    disc = math.exp(-rate * dt)

    # Terminal layer (t = N): spot at node j (j up-moves) = spot * u^j * d^(N-j).
    j = np.arange(steps + 1)
    spot_terminal = spot * u**j * d ** (steps - j)
    values = _intrinsic(spot_terminal, strike, option_type)

    # Backward induction. At layer t we have t+1 nodes; the discounted risk-neutral
    # expectation of the two children gives the continuation value.
    for t in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            jj = np.arange(t + 1)
            spot_level = spot * u**jj * d ** (t - jj)
            values = np.maximum(values, _intrinsic(spot_level, strike, option_type))

    return float(values[0])


def _default_bs_pricer() -> Pricer:
    """Lazily import the Black-Scholes pricer from src.bs for the convergence harness.

    Decoupled on purpose: the harness only needs *a* `Pricer` implementing the frozen
    protocol. We do not hard-code SA-bs's class name at import time; we probe for the
    conventional exports and fall back with a clear error so a caller can always inject
    their own reference pricer instead (see `convergence_order`).
    """
    import src.bs as bs  # local import: src.bs is owned/populated by SA-bs

    for name in ("BlackScholes", "BlackScholesPricer", "BSPricer", "Pricer"):
        candidate = getattr(bs, name, None)
        if candidate is not None:
            return candidate()
    raise ImportError(
        "src.bs does not expose a recognized Black-Scholes Pricer class "
        "(tried BlackScholes/BlackScholesPricer/BSPricer/Pricer). "
        "Pass bs_pricer=<your Pricer> to convergence_order() explicitly."
    )


def convergence_order(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType,
    carry: float = 0.0,
    bs_pricer: Pricer | None = None,
    steps: tuple[int, ...] = (50, 100, 200, 400, 800, 1600),
) -> dict[str, object]:
    """Measure the empirical convergence order of European CRR toward Black-Scholes.

    Prices the European option on the CRR tree at a ladder of step counts `steps`,
    compares each to the Black-Scholes reference price, and fits a line to
    log|CRR(N) - BS| vs log(N). The negated slope is the measured order.

    CRR error is O(1/N) but oscillates with the parity of N (the classic sawtooth),
    so the ladder uses EVEN N only to sample a clean branch of the envelope and get
    a stable ~1.0 slope. See config/tolerances.py['crr_convergence_order_min'].

    Parameters
    ----------
    bs_pricer:
        A Black-Scholes `Pricer` (frozen protocol). If None, lazily imported from
        src.bs. Injectable so this harness never hard-depends on another module's
        class name and can be unit-tested against an independent reference.
    steps:
        Ladder of tree sizes; must all be even for a clean fit.

    Returns
    -------
    dict with keys:
        order            : measured convergence order (float; ~1.0 for correct CRR)
        ns               : the step ladder actually used (even values)
        errors           : |CRR(N) - BS| at each N
        crr_prices       : CRR price at each N
        bs_price         : the Black-Scholes reference price
        log_ns, log_errs : the fitted log-log data
    """
    if bs_pricer is None:
        bs_pricer = _default_bs_pricer()

    ns = tuple(n for n in steps if n % 2 == 0)
    if len(ns) < 2:
        raise ValueError("need >= 2 even step counts to fit a convergence line")

    bs_price = float(
        bs_pricer.price(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=rate,
            sigma=sigma,
            option_type=option_type,
            carry=carry,
        ).price
    )

    crr_prices: list[float] = []
    errors: list[float] = []
    for n in ns:
        crr = _crr_price(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=rate,
            sigma=sigma,
            option_type=option_type,
            carry=carry,
            steps=n,
            american=False,
        )
        crr_prices.append(crr)
        errors.append(abs(crr - bs_price))

    err_arr = np.asarray(errors, dtype=float)
    # Guard against a log(0) if the tree hits BS exactly at some N (rare); replace
    # any zero error with the smallest positive error so the fit stays finite.
    positive = err_arr[err_arr > 0.0]
    if positive.size == 0:
        raise ValueError("CRR matched BS exactly at every N — cannot fit an order")
    err_arr = np.where(err_arr > 0.0, err_arr, positive.min())

    log_ns = np.log(np.asarray(ns, dtype=float))
    log_errs = np.log(err_arr)
    slope, _intercept = np.polyfit(log_ns, log_errs, 1)
    order = float(-slope)

    return {
        "order": order,
        "ns": ns,
        "errors": errors,
        "crr_prices": crr_prices,
        "bs_price": bs_price,
        "log_ns": log_ns.tolist(),
        "log_errs": log_errs.tolist(),
    }
