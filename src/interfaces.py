"""Frozen engine interfaces for vol-lab (the shared protocols; kept stable across the codebase).

Every pricing engine implements `Pricer` (and, where it has Greeks, `GreeksEngine`)
so the cross-engine differential and three-way Greeks reconciliation can treat
Black-Scholes, the CRR lattice, Monte Carlo, and LSMC through one uniform surface.

Frozen numerical conventions (rationale in docs/DESIGN.md):
  * All engines take the SAME primitive inputs so differentials are apples-to-apples:
      spot, strike, tau (year fraction, ACT/365), rate r, carry/dividend q, sigma.
    Under Deribit's inverse-contract, coin-margined convention we price on the
    parity-inferred forward with r = q = 0 in USD terms (documented in DESIGN.md);
    the general r, q signature is retained so the engines are textbook-correct and
    unit-testable against hand-computed cases with nonzero rates.
  * option_type is "C" or "P" (see schema.OptionType).
  * Prices, Greeks, and IVs are in absolute (per-unit-notional) terms; IV is a decimal.

`Greeks` field conventions (the units each sensitivity is reported in):
  * delta   : dPrice/dSpot
  * gamma   : d2Price/dSpot2
  * vega    : dPrice/dSigma      (per 1.00 vol, i.e. per 100 vol-points; scale as needed)
  * theta   : dPrice/dt          (per year; calendar decay is negative for long options)
  * rho     : dPrice/dRate       (per 1.00 rate)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.schema import OptionType

# Array-like inputs to the calibrator (scalar/sequence/ndarray of floats).
ArrayLike = npt.NDArray[np.float64] | Sequence[float] | float


@dataclass(frozen=True, slots=True)
class Greeks:
    """First/second-order sensitivities in the units documented in this module."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class PriceResult:
    """A price plus optional Monte-Carlo-style uncertainty.

    `stderr` and `ci95` are populated by stochastic engines (MC/LSMC) and are None
    for deterministic engines (closed form, lattice). The cross-engine gate uses
    `ci95` to assert the MC interval covers the closed-form price.
    """

    price: float
    stderr: float | None = None
    ci95: tuple[float, float] | None = None


@runtime_checkable
class Pricer(Protocol):
    """Anything that can price a European (and possibly American) vanilla option."""

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
        """Return the option value. `carry` is the cost-of-carry / dividend yield q."""
        ...


@runtime_checkable
class GreeksEngine(Protocol):
    """Anything that can produce Greeks for a European vanilla option."""

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
        ...


@runtime_checkable
class IVSolver(Protocol):
    """Invert a European price to its Black-Scholes implied volatility.

    Implementations must be robust: bracket + Newton with a vega floor, documented
    failure behavior on deep-OTM / near-expiry quotes (return None rather than a
    fabricated value — honest-unknown over plausible-wrong).
    """

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
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Fit a smile/surface parameterization (e.g. SVI) to a set of (k, w) points."""

    def calibrate(
        self,
        *,
        log_moneyness: ArrayLike,   # ln(strike/forward)
        total_variance: ArrayLike,  # w = sigma^2 * tau
        weights: ArrayLike | None = None,  # optional fit weights (e.g. by liquidity)
    ) -> dict[str, float]:
        """Return fitted parameters (raw-SVI: a, b, rho, m, sigma)."""
        ...
