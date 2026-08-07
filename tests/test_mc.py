"""Tests for the Monte Carlo European pricer (SA-mc owned).

The task spec asks these to compare against `src.bs`. As of this wave src/bs is still
an empty package (SA-bs not yet merged), so this file carries a SELF-CONTAINED
closed-form Black-Scholes-Merton reference (`_bs_price` / `_bs_greeks`) built straight
from the textbook formulas. When SA-bs lands, ORCH can swap these helpers for imports
from src.bs; the reference values are identical by construction. This keeps the MC
suite runnable and honestly verified today rather than blocked on a sibling module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from config.tolerances import TOL
from src.interfaces import Greeks, Pricer, PriceResult
from src.mc.engine import (
    DEFAULT_SEED,
    MCGreek,
    MonteCarloPricer,
    variance_reduction_report,
)
from src.schema import OptionType

# --------------------------------------------------------------------------- reference
# Closed-form Black-Scholes-Merton with continuous carry/dividend q (carry == q).
# Conventions match src/interfaces.py: forward F = S*exp((r-q)*tau), df = exp(-r*tau),
# vega per 1.00 vol, theta = dP/dt per year, rho = dP/dRate per 1.00 rate.


def _d1_d2(spot, strike, tau, rate, sigma, carry):
    vol = sigma * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate - carry + 0.5 * sigma * sigma) * tau) / vol
    return d1, d1 - vol


def _bs_price(*, spot, strike, tau, rate, sigma, option_type: OptionType, carry=0.0) -> float:
    d1, d2 = _d1_d2(spot, strike, tau, rate, sigma, carry)
    disc_q = math.exp(-carry * tau)
    disc_r = math.exp(-rate * tau)
    if option_type == "C":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def _bs_greeks(*, spot, strike, tau, rate, sigma, option_type: OptionType, carry=0.0) -> Greeks:
    d1, d2 = _d1_d2(spot, strike, tau, rate, sigma, carry)
    disc_q = math.exp(-carry * tau)
    disc_r = math.exp(-rate * tau)
    pdf = norm.pdf(d1)
    gamma = disc_q * pdf / (spot * sigma * math.sqrt(tau))
    vega = spot * disc_q * pdf * math.sqrt(tau)
    common_theta = -spot * disc_q * pdf * sigma / (2.0 * math.sqrt(tau))
    if option_type == "C":
        delta = disc_q * norm.cdf(d1)
        rho = strike * tau * disc_r * norm.cdf(d2)
        theta = (
            common_theta
            + carry * spot * disc_q * norm.cdf(d1)
            - rate * strike * disc_r * norm.cdf(d2)
        )
    else:
        delta = -disc_q * norm.cdf(-d1)
        rho = -strike * tau * disc_r * norm.cdf(-d2)
        theta = (
            common_theta
            - carry * spot * disc_q * norm.cdf(-d1)
            + rate * strike * disc_r * norm.cdf(-d2)
        )
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# Base case used across several tests (unit-ish spot, ATM-ish, textbook nonzero rate).
BASE = {
    "spot": 100.0, "strike": 100.0, "tau": 0.75, "rate": 0.03, "sigma": 0.65, "carry": 0.01,
}


# --------------------------------------------------------------------------- protocol
def test_implements_pricer_protocol():
    assert isinstance(MonteCarloPricer(), Pricer)


def test_price_returns_priceresult_with_ci():
    pricer = MonteCarloPricer(n_paths=50_000, seed=DEFAULT_SEED)
    res = pricer.price(option_type="C", **BASE)
    assert isinstance(res, PriceResult)
    assert res.stderr is not None and res.stderr > 0.0
    assert res.ci95 is not None
    lo, hi = res.ci95
    # CI is exactly price +/- 1.96*stderr.
    assert lo == pytest.approx(res.price - 1.96 * res.stderr, rel=0, abs=1e-12)
    assert hi == pytest.approx(res.price + 1.96 * res.stderr, rel=0, abs=1e-12)


# --------------------------------------------------------------------------- determinism
def test_determinism_bitwise():
    a = MonteCarloPricer(n_paths=40_000, seed=777).price(option_type="C", **BASE)
    b = MonteCarloPricer(n_paths=40_000, seed=777).price(option_type="C", **BASE)
    # Identical seed -> identical result, bit for bit.
    assert a.price == b.price
    assert a.stderr == b.stderr
    assert a.ci95 == b.ci95


def test_different_seed_differs():
    a = MonteCarloPricer(n_paths=40_000, seed=1).price(option_type="C", **BASE)
    b = MonteCarloPricer(n_paths=40_000, seed=2).price(option_type="C", **BASE)
    assert a.price != b.price


# --------------------------------------------------------------------------- degenerate
def test_zero_tau_is_discounted_intrinsic():
    res = MonteCarloPricer().price(
        spot=120.0, strike=100.0, tau=0.0, rate=0.03, sigma=0.6, option_type="C"
    )
    assert res.price == pytest.approx(20.0, abs=1e-12)
    assert res.stderr == 0.0
    assert res.ci95 == (20.0, 20.0)


def test_zero_sigma_is_deterministic_forward_payoff():
    # sigma=0: S_T = forward exactly; call payoff = df*max(F-K,0).
    spot, strike, tau, rate, carry = 100.0, 90.0, 0.5, 0.04, 0.0
    forward = spot * math.exp((rate - carry) * tau)
    df = math.exp(-rate * tau)
    expected = df * max(forward - strike, 0.0)
    res = MonteCarloPricer().price(
        spot=spot, strike=strike, tau=tau, rate=rate, sigma=0.0, option_type="C"
    )
    assert res.price == pytest.approx(expected, abs=1e-12)
    assert res.stderr == 0.0


# --------------------------------------------------------------------------- accuracy: CI coverage
def test_ci_covers_closed_form_across_grid():
    """Across a strike/vol grid, the fraction of 95% CIs that cover the closed-form BS
    price must exceed TOL['mc_ci_coverage_prob'].

    Statistical-design note: each grid point gets a UNIQUE seed so the trials are
    independent experiments. Reusing one seed across grid points would share the same
    normal draws, collapsing 70 "trials" into ~1 correlated experiment whose luck
    (a single unusually-high draw) would move every point the same direction and make
    the empirical coverage meaningless. Independent seeds is the honest coverage test.
    """
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    vols = [0.30, 0.45, 0.60, 0.75, 0.90]
    covered = 0
    total = 0
    seed = 2000
    for otype in ("C", "P"):
        for k in strikes:
            for sig in vols:
                seed += 1
                res = MonteCarloPricer(n_paths=40_000, seed=seed).price(
                    spot=100.0, strike=k, tau=0.5, rate=0.02, sigma=sig,
                    option_type=otype,
                )
                ref = _bs_price(
                    spot=100.0, strike=k, tau=0.5, rate=0.02, sigma=sig,
                    option_type=otype,
                )
                lo, hi = res.ci95
                if lo <= ref <= hi:
                    covered += 1
                total += 1
    coverage = covered / total
    assert coverage >= TOL["mc_ci_coverage_prob"].value, (
        f"MC CI coverage {coverage:.3f} < {TOL['mc_ci_coverage_prob'].value} "
        f"({covered}/{total})"
    )


def test_ci_halfwidth_within_tol_at_atm():
    """95% CI half-width <= mc_ci_halfwidth_rel of the ATM price at the path count."""
    pricer = MonteCarloPricer(n_paths=200_000, seed=DEFAULT_SEED)
    res = pricer.price(option_type="C", **BASE)
    halfwidth = 1.96 * res.stderr
    rel = halfwidth / res.price
    assert rel <= TOL["mc_ci_halfwidth_rel"].value, (
        f"half-width rel {rel:.4f} > {TOL['mc_ci_halfwidth_rel'].value}"
    )


def test_price_matches_closed_form_within_ci():
    """Point check: the reduced-variance price lands within ~4 stderr of closed form."""
    pricer = MonteCarloPricer(n_paths=200_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        res = pricer.price(option_type=otype, **BASE)
        ref = _bs_price(option_type=otype, **BASE)
        assert abs(res.price - ref) <= 4.0 * res.stderr, (
            f"{otype}: |{res.price} - {ref}| > 4*{res.stderr}"
        )


# --------------------------------------------------------------------------- variance reduction
def test_variance_reduction_speedup_gt_one():
    rep = variance_reduction_report(option_type="C", n_paths=100_000, **BASE)
    # Each technique should reduce estimator variance; the full combo strictly so.
    assert rep["speedup_antithetic"] > 1.0, rep
    assert rep["speedup_control"] > 1.0, rep
    assert rep["speedup_full"] > 1.0, rep
    # Antithetic-reduced stderr must be smaller than plain.
    assert rep["stderr_full"] < rep["stderr_plain"]


# --------------------------------------------------------------------------- Greeks vs closed form
def _assert_greek(mc: MCGreek, ref: float, name: str):
    tol = TOL["greeks_mc_vs_closed_rel"].value
    rel = abs(mc.value - ref) / abs(ref)
    assert rel <= tol, (
        f"{name} [{mc.method}]: mc={mc.value:.6f} ref={ref:.6f} "
        f"rel={rel:.4f} > {tol} (stderr={mc.stderr:.6f})"
    )


def test_pathwise_delta_vs_closed_form():
    pricer = MonteCarloPricer(n_paths=400_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        ref = _bs_greeks(option_type=otype, **BASE)
        mc = pricer.pathwise_delta(option_type=otype, **BASE)
        _assert_greek(mc, ref.delta, f"delta {otype}")


def test_pathwise_vega_vs_closed_form():
    pricer = MonteCarloPricer(n_paths=400_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        ref = _bs_greeks(option_type=otype, **BASE)
        mc = pricer.pathwise_vega(option_type=otype, **BASE)
        _assert_greek(mc, ref.vega, f"vega {otype}")


def test_lr_delta_vs_closed_form():
    pricer = MonteCarloPricer(n_paths=400_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        ref = _bs_greeks(option_type=otype, **BASE)
        mc = pricer.lr_delta(option_type=otype, **BASE)
        _assert_greek(mc, ref.delta, f"lr-delta {otype}")


def test_lr_gamma_vs_closed_form():
    # Gamma is a small number; use extra paths for the higher-variance LR estimator.
    pricer = MonteCarloPricer(n_paths=800_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        ref = _bs_greeks(option_type=otype, **BASE)
        mc = pricer.lr_gamma(option_type=otype, **BASE)
        _assert_greek(mc, ref.gamma, f"lr-gamma {otype}")


def test_pathwise_rho_theta_vs_closed_form():
    # rho/theta are the extra pathwise estimators surfaced through greeks(); verify them.
    pricer = MonteCarloPricer(n_paths=400_000, seed=DEFAULT_SEED)
    for otype in ("C", "P"):
        ref = _bs_greeks(option_type=otype, **BASE)
        _assert_greek(pricer.pathwise_rho(option_type=otype, **BASE), ref.rho, f"rho {otype}")
        _assert_greek(
            pricer.pathwise_theta(option_type=otype, **BASE), ref.theta, f"theta {otype}"
        )


def test_greeks_object_assembled():
    pricer = MonteCarloPricer(n_paths=400_000, seed=DEFAULT_SEED)
    g = pricer.greeks(option_type="C", **BASE)
    ref = _bs_greeks(option_type="C", **BASE)
    tol = TOL["greeks_mc_vs_closed_rel"].value
    assert abs(g.delta - ref.delta) / abs(ref.delta) <= tol
    assert abs(g.vega - ref.vega) / abs(ref.vega) <= tol
    assert abs(g.gamma - ref.gamma) / abs(ref.gamma) <= tol


def test_greeks_raise_on_degenerate():
    pricer = MonteCarloPricer(n_paths=1000, seed=1)
    with pytest.raises(ValueError):
        pricer.pathwise_delta(spot=100, strike=100, tau=0.0, rate=0.0, sigma=0.5, option_type="C")
    with pytest.raises(ValueError):
        pricer.lr_gamma(spot=100, strike=100, tau=0.5, rate=0.0, sigma=0.0, option_type="C")


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        MonteCarloPricer(n_paths=1000).price(
            spot=100, strike=100, tau=0.5, rate=0.0, sigma=0.5, option_type="X"  # type: ignore[arg-type]
        )


def test_antithetic_odd_paths_rounded_even():
    p = MonteCarloPricer(n_paths=9999, antithetic=True)
    assert p.n_paths == 10_000


def test_stderr_scaling_sanity():
    """Plain-MC stderr should fall ~1/sqrt(n): 4x paths -> ~2x tighter."""
    kw = dict(option_type="C", **BASE)
    se_small = MonteCarloPricer(
        n_paths=50_000, seed=DEFAULT_SEED, antithetic=False, control_variate=False
    ).price(**kw).stderr
    se_big = MonteCarloPricer(
        n_paths=200_000, seed=DEFAULT_SEED, antithetic=False, control_variate=False
    ).price(**kw).stderr
    ratio = se_small / se_big
    assert 1.7 <= ratio <= 2.3, f"stderr ratio {ratio:.3f} not ~2"


def test_numpy_available():
    # guard: engine relies on numpy vectorization
    assert np.__version__
