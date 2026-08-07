"""Monte Carlo European option pricer (G1 — owned by SA-mc).

Seeded, deterministic GBM terminal-price simulation for European vanillas. Because
the payoff of a European option depends only on the *terminal* spot, we sample
S_T with a single exact lognormal step to tau (no path discretization, no Euler
bias) — the simulation is exact for GBM and the only error is Monte-Carlo noise.

Numerical conventions match the frozen contracts (src/interfaces.py):
  * Forward  F  = spot * exp((rate - carry) * tau)
  * Discount df = exp(-rate * tau)
  * option_type is "C" or "P"; sigma/IV are decimals; tau is a year fraction.
  * carry = cost-of-carry / dividend yield q; the risk-neutral drift is (rate - carry).

Variance reduction (both toggleable in the constructor so `price` keeps the exact
frozen Pricer signature):
  * ANTITHETIC variates: draw m standard normals Z and reuse -Z, halving the number
    of independent draws while preserving the mean; variance is reported on the m
    antithetic *pair means* so the stderr stays statistically honest.
  * CONTROL VARIATE: the discounted terminal spot  df * S_T  has the known exact
    expectation  E[df * S_T] = spot * exp(-carry * tau)  and is strongly correlated
    with the payoff. We subtract the optimal-coefficient-scaled control deviation;
    the estimator stays unbiased because the control's mean is known analytically.

Greeks (each estimator reports its OWN standard error):
  * PATHWISE delta, vega (also rho, theta) — differentiate the payoff through S_T;
    low variance, but only valid because the vanilla payoff is a.s. differentiable.
  * LIKELIHOOD-RATIO delta and gamma — differentiate the lognormal density; higher
    variance but works for discontinuous payoffs and second order (gamma), where the
    pathwise estimator of the *undifferentiated* payoff would fail.

Failure modes (honest-unknown over plausible-wrong):
  * tau <= 0 -> deterministic discounted intrinsic, stderr 0.0 (no randomness).
  * sigma <= 0 -> deterministic (degenerate) forward payoff, stderr 0.0.
  * Greeks require tau > 0 and sigma > 0 (the estimator weights divide by
    sigma*sqrt(tau)); they raise ValueError otherwise rather than return a NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.common import DEFAULT_SEED
from src.common import mean_stderr as _mean_stderr
from src.interfaces import Greeks, PriceResult
from src.schema import OptionType

DEFAULT_PATHS = 200_000


@dataclass(frozen=True, slots=True)
class MCGreek:
    """A single Monte-Carlo Greek estimate with its own sampling standard error.

    `method` names the estimator ("pathwise" / "likelihood_ratio") so a reconciliation
    report can attribute the noise to the technique that produced it.
    """

    value: float
    stderr: float
    method: str


def _payoff(terminal: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    """Undiscounted European vanilla payoff."""
    if option_type == "C":
        return np.maximum(terminal - strike, 0.0)
    if option_type == "P":
        return np.maximum(strike - terminal, 0.0)
    raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")


class MonteCarloPricer:
    """Seeded GBM Monte-Carlo pricer implementing Pricer and GreeksEngine.

    Variance-reduction switches live on the instance so `price`/`greeks` keep the
    exact frozen keyword signature; construct different instances to compare plain
    vs variance-reduced runs at the same seed.
    """

    def __init__(
        self,
        *,
        n_paths: int = DEFAULT_PATHS,
        seed: int = DEFAULT_SEED,
        antithetic: bool = True,
        control_variate: bool = True,
    ) -> None:
        if n_paths < 2:
            raise ValueError("n_paths must be >= 2")
        # Antithetic sampling pairs Z with -Z, so we need an even count; round up by
        # one path if needed (documented; keeps determinism obvious).
        if antithetic and n_paths % 2 == 1:
            n_paths += 1
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.control_variate = control_variate

    # ------------------------------------------------------------------ simulation
    def _draw_normals(self, rng: np.random.Generator) -> np.ndarray:
        """Standard normals; antithetic layout is [Z_1..Z_m, -Z_1..-Z_m]."""
        if self.antithetic:
            m = self.n_paths // 2
            half = rng.standard_normal(m)
            return np.concatenate([half, -half])
        return rng.standard_normal(self.n_paths)

    def _terminal(
        self,
        rng: np.random.Generator,
        *,
        spot: float,
        tau: float,
        rate: float,
        sigma: float,
        carry: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact lognormal step to tau: returns (S_T array, Z array)."""
        z = self._draw_normals(rng)
        drift = (rate - carry - 0.5 * sigma * sigma) * tau
        diffusion = sigma * math.sqrt(tau) * z
        terminal = spot * np.exp(drift + diffusion)
        return terminal, z

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
        """Monte-Carlo European price with stderr and a 95% CI (price +/- 1.96*stderr)."""
        df = math.exp(-rate * tau)

        # Degenerate regimes: no randomness -> exact discounted (intrinsic/forward) value.
        if tau <= 0.0 or sigma <= 0.0:
            forward = spot * math.exp((rate - carry) * tau) if tau > 0.0 else spot
            intrinsic = float(_payoff(np.array([forward]), strike, option_type)[0])
            price = df * intrinsic
            return PriceResult(price=price, stderr=0.0, ci95=(price, price))

        rng = np.random.default_rng(self.seed)
        terminal, _z = self._terminal(
            rng, spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        disc_payoff = df * _payoff(terminal, strike, option_type)

        if self.control_variate:
            # Control = discounted terminal spot, exact mean E[df*S_T] = spot*exp(-carry*tau).
            control = df * terminal
            control_mean = spot * math.exp(-carry * tau)
            var_c = control.var(ddof=1)
            if var_c > 0.0:
                cov = np.cov(disc_payoff, control, ddof=1)[0, 1]
                beta = cov / var_c
            else:
                beta = 0.0
            estimator = disc_payoff - beta * (control - control_mean)
        else:
            estimator = disc_payoff

        price, stderr = _mean_stderr(estimator, antithetic=self.antithetic)
        ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)
        return PriceResult(price=price, stderr=stderr, ci95=ci95)

    # ------------------------------------------------------------------ greeks
    def _require_regular(self, tau: float, sigma: float) -> None:
        if tau <= 0.0 or sigma <= 0.0:
            raise ValueError("MC Greeks require tau > 0 and sigma > 0")

    def _sim_for_greeks(
        self,
        *,
        spot: float,
        tau: float,
        rate: float,
        sigma: float,
        carry: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Shared draw for every Greek estimator; same seed -> identical (S_T, Z, df)."""
        rng = np.random.default_rng(self.seed)
        terminal, z = self._terminal(
            rng, spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        df = math.exp(-rate * tau)
        return terminal, z, df

    def pathwise_delta(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Pathwise dP/dSpot: differentiate payoff through dS_T/dSpot = S_T/spot."""
        self._require_regular(tau, sigma)
        terminal, _z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        ratio = terminal / spot
        if option_type == "C":
            g = df * ratio * (terminal > strike)
        else:
            g = -df * ratio * (terminal < strike)
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "pathwise")

    def pathwise_vega(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Pathwise dP/dSigma: dS_T/dSigma = S_T*(sqrt(tau)*Z - sigma*tau)."""
        self._require_regular(tau, sigma)
        terminal, z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        dst_dsigma = terminal * (math.sqrt(tau) * z - sigma * tau)
        if option_type == "C":
            g = df * (terminal > strike) * dst_dsigma
        else:
            g = -df * (terminal < strike) * dst_dsigma
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "pathwise")

    def pathwise_rho(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Pathwise dP/dRate: both the discount df and the drift depend on rate."""
        self._require_regular(tau, sigma)
        terminal, _z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        payoff = _payoff(terminal, strike, option_type)
        # dS_T/drate = tau * S_T; d(payoff)/drate = indicator * tau * S_T (sign by type).
        if option_type == "C":
            indicator = (terminal > strike).astype(float)
        else:
            indicator = -(terminal < strike).astype(float)
        dpayoff = indicator * tau * terminal
        g = -tau * df * payoff + df * dpayoff
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "pathwise")

    def pathwise_theta(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Pathwise theta = dP/dt = -dP/dtau (per year; negative for long options)."""
        self._require_regular(tau, sigma)
        terminal, z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        payoff = _payoff(terminal, strike, option_type)
        # dS_T/dtau = S_T * [(rate - carry - 0.5*sigma^2) + sigma*Z/(2*sqrt(tau))].
        dst_dtau = terminal * (
            (rate - carry - 0.5 * sigma * sigma) + sigma * z / (2.0 * math.sqrt(tau))
        )
        if option_type == "C":
            indicator = (terminal > strike).astype(float)
        else:
            indicator = -(terminal < strike).astype(float)
        dpayoff = indicator * dst_dtau
        dp_dtau = -rate * df * payoff + df * dpayoff
        g = -dp_dtau  # theta is decay in calendar time
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "pathwise")

    def lr_delta(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Likelihood-ratio dP/dSpot: score weight Z/(spot*sigma*sqrt(tau))."""
        self._require_regular(tau, sigma)
        terminal, z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        payoff = _payoff(terminal, strike, option_type)
        weight = z / (spot * sigma * math.sqrt(tau))
        g = df * payoff * weight
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "likelihood_ratio")

    def lr_gamma(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        sigma: float,
        option_type: OptionType,
        carry: float = 0.0,
    ) -> MCGreek:
        """Likelihood-ratio d2P/dSpot2.

        Second-order score weight for GBM (Glasserman, MCMFE eq. 7.35):
            (Z^2 - 1 - sigma*sqrt(tau)*Z) / (spot^2 * sigma^2 * tau).
        """
        self._require_regular(tau, sigma)
        terminal, z, df = self._sim_for_greeks(
            spot=spot, tau=tau, rate=rate, sigma=sigma, carry=carry
        )
        payoff = _payoff(terminal, strike, option_type)
        s_sqrt_t = sigma * math.sqrt(tau)
        weight = (z * z - 1.0 - s_sqrt_t * z) / (spot * spot * sigma * sigma * tau)
        g = df * payoff * weight
        value, stderr = _mean_stderr(g, antithetic=self.antithetic)
        return MCGreek(value, stderr, "likelihood_ratio")

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
        """Assemble a full Greeks object from the MC estimators (GreeksEngine contract).

        delta/vega/theta/rho use the low-variance pathwise estimator; gamma uses the
        likelihood-ratio estimator (pathwise fails for the second derivative of a
        payoff with a kink). Per-estimator stderrs are available via the dedicated
        methods above for the honest reconciliation report.
        """
        # Each keyword is passed explicitly (not **splatted from a mixed-type dict) so the
        # static types of option_type (Literal) and the floats are preserved.
        return Greeks(
            delta=self.pathwise_delta(
                spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
                option_type=option_type, carry=carry).value,
            gamma=self.lr_gamma(
                spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
                option_type=option_type, carry=carry).value,
            vega=self.pathwise_vega(
                spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
                option_type=option_type, carry=carry).value,
            theta=self.pathwise_theta(
                spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
                option_type=option_type, carry=carry).value,
            rho=self.pathwise_rho(
                spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
                option_type=option_type, carry=carry).value,
        )


def variance_reduction_report(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType,
    carry: float = 0.0,
    n_paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Measure the variance-reduction speedup at equal paths and seed.

    The 'speedup' is the ratio of estimator variances (stderr^2) of plain MC to the
    fully variance-reduced estimator at the SAME path count and seed. Because the
    stderr of a mean scales as 1/sqrt(n), this ratio is exactly the factor by which
    plain MC would have to increase its path count to match the reduced estimator's
    stderr — i.e. the equal-precision path-count multiplier.
    """
    def _run(antithetic: bool, control: bool) -> float:
        p = MonteCarloPricer(
            n_paths=n_paths, seed=seed, antithetic=antithetic, control_variate=control
        ).price(
            spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma,
            option_type=option_type, carry=carry,
        )
        return p.stderr if p.stderr is not None else 0.0

    se_plain = _run(False, False)
    se_anti = _run(True, False)
    se_ctrl = _run(False, True)
    se_full = _run(True, True)

    def _speedup(se: float) -> float:
        return (se_plain / se) ** 2 if se > 0.0 else float("inf")

    return {
        "n_paths": float(n_paths),
        "stderr_plain": se_plain,
        "stderr_antithetic": se_anti,
        "stderr_control": se_ctrl,
        "stderr_full": se_full,
        "speedup_antithetic": _speedup(se_anti),
        "speedup_control": _speedup(se_ctrl),
        "speedup_full": _speedup(se_full),
    }
