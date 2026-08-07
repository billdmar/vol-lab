"""Longstaff-Schwartz Monte-Carlo (LSMC) American option pricer (G3 — owned by SA-lsmc).

Seeded, deterministic GBM path simulation + least-squares regression of the
continuation value on in-the-money paths (Longstaff & Schwartz, RFS 2001). This
is the Monte-Carlo reference for American vanillas, cross-checked against a fine
CRR lattice (see config/tolerances.py['lsmc_vs_crr_american_rel']).

Numerical conventions match the frozen contracts (src/interfaces.py):
  * Risk-neutral drift (rate - carry); discount df = exp(-rate * tau).
  * option_type is "C" or "P"; sigma/IV are decimals; tau is a year fraction.
  * We simulate FULL paths (unlike the European MC engine, which needs only S_T)
    because early exercise is path/time dependent. Each step is an EXACT lognormal
    increment, so there is no Euler discretization bias — the only path-level error
    is the discreteness of the exercise dates (n_steps early-exercise opportunities
    approximating the continuous American right), plus regression + MC noise.

The Longstaff-Schwartz recursion (cashflow bookkeeping form):
  1. Initialize each path's cashflow to the terminal intrinsic at step N.
  2. Walk backward t = N-1 .. 1. On the IN-THE-MONEY paths only, regress the
     discounted realized future cashflow on a polynomial basis of the (scaled)
     spot; the fitted value is the estimated continuation value. Exercise where
     the immediate intrinsic exceeds that continuation, overwriting the path's
     cashflow and its exercise time.
  3. The price is the mean over paths of the cashflow discounted from its
     exercise time back to today; step 0 (t=0) is never an exercise date because
     the holder does not exercise at inception for a positive-time option.

Regression on ITM paths only is the standard variance/robustness choice: OTM
paths carry no exercise decision, and including them wastes basis degrees of
freedom fitting a region where continuation == the whole (already correct)
discounted value.

Variance reduction / determinism:
  * ANTITHETIC normals (Z, -Z) per step reduce variance and are reported as pair
    means so the stderr stays statistically honest (same convention as src/mc).
  * A fixed seed makes the price bit-for-bit reproducible.

Failure modes (honest-unknown over plausible-wrong):
  * tau <= 0 -> American value is the immediate intrinsic (exercise now); stderr 0.
  * sigma <= 0 -> degenerate deterministic forward path; American value is the best
    of exercising along a riskless trajectory. We fall back to the intrinsic of the
    deterministic forward at each step and take the max, discounted — stderr 0.
"""

from __future__ import annotations

import math

import numpy as np

from src.common import DEFAULT_SEED
from src.common import intrinsic as _intrinsic
from src.common import mean_stderr as _mean_stderr
from src.interfaces import PriceResult
from src.schema import OptionType

DEFAULT_PATHS = 100_000
DEFAULT_STEPS = 50
DEFAULT_BASIS_DEGREE = 3


class LSMCPricer:
    """Seeded Longstaff-Schwartz Monte-Carlo pricer for American vanilla options.

    Parameters
    ----------
    n_paths:
        Number of simulated GBM paths. With antithetic sampling this is rounded up
        to the next even number so the (Z, -Z) pairing is exact.
    n_steps:
        Number of time steps = number of discrete early-exercise dates. More steps
        approach the continuous-exercise American price (from below).
    seed:
        RNG seed; a fixed seed makes the price bit-for-bit reproducible.
    basis_degree:
        Degree of the simple-power regression basis {1, x, x^2, ..., x^d} used for
        the continuation-value fit. Degree 3 is the Longstaff-Schwartz default and
        is ample for a single-asset vanilla.
    antithetic:
        Toggle antithetic variates (default True).
    """

    def __init__(
        self,
        *,
        n_paths: int = DEFAULT_PATHS,
        n_steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        basis_degree: int = DEFAULT_BASIS_DEGREE,
        antithetic: bool = True,
    ) -> None:
        if n_paths < 2:
            raise ValueError("n_paths must be >= 2")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if basis_degree < 1:
            raise ValueError("basis_degree must be >= 1")
        # Antithetic sampling pairs Z with -Z, so we need an even count; round up by
        # one path if needed (documented; keeps determinism obvious).
        if antithetic and n_paths % 2 == 1:
            n_paths += 1
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.basis_degree = basis_degree
        self.antithetic = antithetic

    # ------------------------------------------------------------------ simulation
    def _simulate_paths(
        self,
        *,
        spot: float,
        tau: float,
        rate: float,
        sigma: float,
        carry: float,
    ) -> np.ndarray:
        """Simulate GBM paths of shape (n_paths, n_steps + 1); column 0 == spot.

        Each step is an exact lognormal increment (no Euler bias). Antithetic layout
        pairs the first half of the paths with the sign-flipped normals of the second
        half at EVERY step, so the pairing survives the whole trajectory.
        """
        n, steps = self.n_paths, self.n_steps
        dt = tau / steps
        drift = (rate - carry - 0.5 * sigma * sigma) * dt
        vol_step = sigma * math.sqrt(dt)

        rng = np.random.default_rng(self.seed)
        if self.antithetic:
            m = n // 2
            half = rng.standard_normal((m, steps))
            z = np.concatenate([half, -half], axis=0)
        else:
            z = rng.standard_normal((n, steps))

        # Cumulative log-returns -> paths. log S_t = log spot + sum of increments.
        increments = drift + vol_step * z
        log_paths = np.cumsum(increments, axis=1) + math.log(spot)
        paths = np.empty((n, steps + 1), dtype=float)
        paths[:, 0] = spot
        paths[:, 1:] = np.exp(log_paths)
        return paths

    # ------------------------------------------------------------------ price
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
        """LSMC American price with stderr and a 95% CI (price +/- 1.96*stderr)."""
        if option_type not in ("C", "P"):
            raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")

        # Degenerate regimes -> no randomness; American value is deterministic.
        if tau <= 0.0:
            price = float(_intrinsic(np.array([spot]), strike, option_type)[0])
            return PriceResult(price=price, stderr=0.0, ci95=(price, price))
        if sigma <= 0.0:
            price = self._deterministic_american(
                spot=spot, strike=strike, tau=tau, rate=rate, carry=carry,
                option_type=option_type,
            )
            return PriceResult(price=price, stderr=0.0, ci95=(price, price))

        paths = self._simulate_paths(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        n, steps = self.n_paths, self.n_steps
        dt = tau / steps
        disc_step = math.exp(-rate * dt)

        # Cashflow bookkeeping: value realized on each path and the step at which it
        # is realized. Initialize with terminal exercise at step N.
        cashflow = _intrinsic(paths[:, steps], strike, option_type)
        exercise_step = np.full(n, steps, dtype=np.int64)

        # Backward induction over the interior exercise dates t = N-1 .. 1.
        for t in range(steps - 1, 0, -1):
            intrinsic = _intrinsic(paths[:, t], strike, option_type)
            itm = intrinsic > 0.0
            if not np.any(itm):
                continue

            # Discount each path's realized future cashflow back to time t.
            disc_future = cashflow[itm] * disc_step ** (exercise_step[itm] - t)

            # Regress continuation on a scaled-spot polynomial basis over ITM paths.
            x = paths[itm, t] / strike  # scale to O(1) for a well-conditioned fit
            design = np.vander(x, self.basis_degree + 1, increasing=True)
            coeffs, *_ = np.linalg.lstsq(design, disc_future, rcond=None)
            continuation = design @ coeffs

            # Exercise where the immediate intrinsic beats the estimated continuation.
            exercise = intrinsic[itm] > continuation
            idx = np.nonzero(itm)[0][exercise]
            cashflow[idx] = intrinsic[idx]
            exercise_step[idx] = t

        # Discount each path's cashflow from its exercise step back to today.
        per_path = cashflow * disc_step ** exercise_step
        price, stderr = _mean_stderr(per_path, antithetic=self.antithetic)
        ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)
        return PriceResult(price=price, stderr=stderr, ci95=ci95)

    # ------------------------------------------------------------------ degenerate
    def _deterministic_american(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        carry: float,
        option_type: OptionType,
    ) -> float:
        """American value on the riskless (sigma=0) forward trajectory.

        With no diffusion the spot follows S_t = spot * exp((rate - carry) * t)
        deterministically. The optimal policy exercises at whichever of the n_steps
        dates maximizes the discounted intrinsic; we take that max over the grid.
        """
        steps = self.n_steps
        t_grid = np.linspace(0.0, tau, steps + 1)
        fwd = spot * np.exp((rate - carry) * t_grid)
        intrinsic = _intrinsic(fwd, strike, option_type)
        disc = np.exp(-rate * t_grid)
        return float(np.max(intrinsic * disc))
