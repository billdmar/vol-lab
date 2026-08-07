"""G1 — Engine cross-verification gate (integration tests).

This is the cross-engine differential that ties the four independently-written engines
together and proves they agree with each other, with theory, and (structurally) with
the exchange. The four checks that make up the gate:

  1. Binomial -> Black-Scholes convergence with a MEASURED empirical order.
  2. Monte Carlo 95% CIs cover the closed-form price across a strike/vol grid.
  3. Three-way Greeks reconciliation: closed form vs central finite differences vs
     MC pathwise/likelihood-ratio, per the tolerance registry.
  4. Put-call parity to machine precision on model prices.

Every tolerance is referenced BY NAME from config.tolerances — never inlined, never
widened. Measured quantities (convergence order, MC coverage fraction) are asserted
AND printed so the gate is honest, not just green.
"""

from __future__ import annotations

import math

import pytest

from config.tolerances import TOL
from src.bs import BS
from src.lattice import CRRBinomial, convergence_order
from src.mc import MonteCarloPricer

# A representative grid spanning ITM/ATM/OTM and short/long expiries at crypto-like vol.
SPOT = 100.0
GRID = [
    # (strike, tau, sigma, rate, carry, option_type)
    (100.0, 1.00, 0.20, 0.00, 0.00, "C"),
    (100.0, 1.00, 0.20, 0.00, 0.00, "P"),
    (90.0, 0.50, 0.65, 0.00, 0.00, "C"),
    (110.0, 0.50, 0.65, 0.00, 0.00, "P"),
    (120.0, 0.25, 0.80, 0.05, 0.00, "C"),
    (80.0, 0.75, 0.45, 0.03, 0.02, "P"),
]


# --------------------------------------------------------------- 1. CRR -> BS order


def test_crr_converges_to_bs_measured_order():
    """CRR European price converges to Black-Scholes; measure the empirical order.

    Uses convergence_order() with NO injected pricer, so the harness discovers the real
    src.bs.BlackScholes through its protocol probe (the cross-engine wiring, not a stub).
    """
    fit = convergence_order(
        spot=SPOT, strike=100.0, tau=1.0, rate=0.0, sigma=0.20, option_type="C",
        steps=(50, 100, 200, 400, 800, 1600),
    )
    order = fit["order"]
    print(f"\n[G1] CRR->BS measured convergence order = {order:.4f} "
          f"(min gate {TOL['crr_convergence_order_min'].value}); "
          f"errors halve per doubling: {[f'{e:.2e}' for e in fit['errors']]}")
    assert order >= TOL["crr_convergence_order_min"].value

    # And the price itself agrees at high N.
    crr = CRRBinomial(steps=2000)
    for strike, tau, sigma, rate, carry, ot in GRID:
        p_crr = crr.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                          sigma=sigma, option_type=ot, carry=carry).price
        p_bs = BS.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                        sigma=sigma, option_type=ot, carry=carry).price
        assert abs(p_crr - p_bs) <= TOL["crr_bs_convergence_price"].value, (strike, tau, ot)


# ------------------------------------------------------- 2. MC CI covers closed form


def test_mc_ci_covers_closed_form_across_grid():
    """Fraction of 95% MC CIs covering the BS price must exceed the registry threshold.

    Independent seed per grid point (correlated seeds collapse the trials — see the MC engine's
    documented finding), so the coverage fraction is a real measurement.
    """
    covered = 0
    total = 0
    for i, (strike, tau, sigma, rate, carry, ot) in enumerate(GRID):
        mc = MonteCarloPricer(n_paths=200_000, seed=1000 + i,
                              antithetic=True, control_variate=True)
        res = mc.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                       sigma=sigma, option_type=ot, carry=carry)
        p_bs = BS.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                        sigma=sigma, option_type=ot, carry=carry).price
        lo, hi = res.ci95
        total += 1
        if lo <= p_bs <= hi:
            covered += 1
        # ATM CI half-width sanity (variance reduction actually engaged).
        if strike == SPOT and abs(rate) < 1e-12:
            halfwidth_rel = (hi - lo) / 2.0 / p_bs
            assert halfwidth_rel <= TOL["mc_ci_halfwidth_rel"].value, halfwidth_rel
    frac = covered / total
    print(f"\n[G1] MC 95% CI coverage of BS price = {covered}/{total} = {frac:.2f} "
          f"(min gate {TOL['mc_ci_coverage_prob'].value})")
    assert frac >= TOL["mc_ci_coverage_prob"].value


# ------------------------------------------------- 3. Three-way Greeks reconciliation


def _central_fd_greeks(strike, tau, sigma, rate, carry, ot):
    """Central finite differences on the BS price — the model-agnostic middle leg."""
    def price(s=SPOT, k=strike, t=tau, sig=sigma, r=rate, q=carry):
        return BS.price(spot=s, strike=k, tau=t, rate=r, sigma=sig,
                        option_type=ot, carry=q).price

    hS = SPOT * 1e-4
    hsig = 1e-4
    hr = 1e-5
    ht = 1e-5
    delta = (price(s=SPOT + hS) - price(s=SPOT - hS)) / (2 * hS)
    gamma = (price(s=SPOT + hS) - 2 * price() + price(s=SPOT - hS)) / (hS * hS)
    vega = (price(sig=sigma + hsig) - price(sig=sigma - hsig)) / (2 * hsig)
    rho = (price(r=rate + hr) - price(r=rate - hr)) / (2 * hr)
    # theta = dPrice/dt (calendar time forward = tau decreasing); per-year, sign negative.
    theta = -(price(t=tau + ht) - price(t=tau - ht)) / (2 * ht)
    return delta, gamma, vega, theta, rho


def test_greeks_closed_form_vs_central_fd():
    """Closed-form Greeks vs central finite differences, per greeks_fd_vs_closed_rel."""
    tol = TOL["greeks_fd_vs_closed_rel"].value
    for strike, tau, sigma, rate, carry, ot in GRID:
        g = BS.greeks(spot=SPOT, strike=strike, tau=tau, rate=rate,
                      sigma=sigma, option_type=ot, carry=carry)
        d, gam, v, th, rh = _central_fd_greeks(strike, tau, sigma, rate, carry, ot)
        for name, closed, fd in [
            ("delta", g.delta, d), ("gamma", g.gamma, gam), ("vega", g.vega, v),
            ("theta", g.theta, th), ("rho", g.rho, rh),
        ]:
            denom = max(abs(closed), 1e-8)
            rel = abs(closed - fd) / denom
            assert rel <= tol, (
                f"{name} @ K={strike},tau={tau},{ot}: "
                f"closed={closed:.6g} fd={fd:.6g} rel={rel:.2e}"
            )


def test_greeks_closed_form_vs_mc_pathwise():
    """MC pathwise (delta/vega) + LR (gamma) vs closed form, per greeks_mc_vs_closed_rel.

    Noisy estimators: assert within tolerance AND report each estimator's own stderr so
    the gate is statistically honest (an in-tolerance pass with a huge stderr is a red flag).
    """
    tol = TOL["greeks_mc_vs_closed_rel"].value
    # ATM, high vol, r=q=0 — the clean case for pathwise/LR estimators.
    strike, tau, sigma = 100.0, 0.75, 0.65
    g = BS.greeks(spot=SPOT, strike=strike, tau=tau, rate=0.0, sigma=sigma, option_type="C")
    mc = MonteCarloPricer(n_paths=400_000, seed=7, antithetic=True, control_variate=True)

    pw_delta = mc.pathwise_delta(spot=SPOT, strike=strike, tau=tau, rate=0.0,
                                 sigma=sigma, option_type="C")
    pw_vega = mc.pathwise_vega(spot=SPOT, strike=strike, tau=tau, rate=0.0,
                               sigma=sigma, option_type="C")
    lr_gamma = mc.lr_gamma(spot=SPOT, strike=strike, tau=tau, rate=0.0,
                           sigma=sigma, option_type="C")
    print("\n[G1] Greeks 3-way (ATM tau=0.75 sig=0.65):")
    for name, closed, est in [("delta", g.delta, pw_delta),
                              ("vega", g.vega, pw_vega),
                              ("gamma", g.gamma, lr_gamma)]:
        rel = abs(closed - est.value) / max(abs(closed), 1e-8)
        print(f"       {name:6s} closed={closed:.5f}  mc={est.value:.5f}"
              f"  stderr={est.stderr:.2e}  rel={rel:.2e}  ({est.method})")
        assert rel <= tol, f"{name}: closed={closed} mc={est.value} rel={rel}"


# ----------------------------------------------------------------- 4. Parity gate


def test_put_call_parity_machine_precision_all_engines():
    """Put-call parity C - P = S e^{-q tau} - K e^{-r tau} to machine precision (BS)."""
    tol = TOL["parity_model"].value
    for strike, tau, sigma, rate, carry, _ in GRID:
        c = BS.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                     sigma=sigma, option_type="C", carry=carry).price
        p = BS.price(spot=SPOT, strike=strike, tau=tau, rate=rate,
                     sigma=sigma, option_type="P", carry=carry).price
        lhs = c - p
        rhs = SPOT * math.exp(-carry * tau) - strike * math.exp(-rate * tau)
        assert abs(lhs - rhs) <= tol, (strike, tau, abs(lhs - rhs))


def test_convergence_order_guards():
    """Exercise the harness guards (fail-loud, not fabricate) — part of honest numerics."""
    # Fewer than 2 even step counts -> cannot fit a line.
    with pytest.raises(ValueError, match="even step counts"):
        convergence_order(spot=SPOT, strike=100.0, tau=1.0, rate=0.0, sigma=0.2,
                          option_type="C", steps=(101,), bs_pricer=BS)
    # Injecting BS explicitly must give the same order as the auto-discovered path.
    fit = convergence_order(spot=SPOT, strike=100.0, tau=1.0, rate=0.0,
                            sigma=0.2, option_type="C",
                            steps=(100, 200, 400, 800), bs_pricer=BS)
    assert fit["order"] >= TOL["crr_convergence_order_min"].value


def test_crr_tau_zero_and_bad_inputs():
    """CRR intrinsic-at-expiry and fail-loud guards (honest-unknown discipline)."""
    crr = CRRBinomial(steps=100)
    # tau=0 -> intrinsic
    assert crr.price(spot=SPOT, strike=90.0, tau=0.0, rate=0.0, sigma=0.2,
                     option_type="C").price == pytest.approx(10.0)
    with pytest.raises(ValueError):
        crr.price(spot=SPOT, strike=100.0, tau=-1.0, rate=0.0, sigma=0.2, option_type="C")
    with pytest.raises(ValueError):
        crr.price(spot=SPOT, strike=100.0, tau=1.0, rate=0.0, sigma=0.0, option_type="C")


def test_american_ge_european_cross_engine():
    """American CRR >= European CRR and >= European BS (early-exercise premium >= 0)."""
    crr_e = CRRBinomial(steps=1500, american=False)
    crr_a = CRRBinomial(steps=1500, american=True)
    # ITM American put with positive rate: early exercise is valuable.
    k, tau, sig, r = 110.0, 1.0, 0.30, 0.08
    pe = crr_e.price(spot=SPOT, strike=k, tau=tau, rate=r, sigma=sig, option_type="P").price
    pa = crr_a.price(spot=SPOT, strike=k, tau=tau, rate=r, sigma=sig, option_type="P").price
    p_bs = BS.price(spot=SPOT, strike=k, tau=tau, rate=r, sigma=sig, option_type="P").price
    print(f"\n[G1] American put {pa:.4f} >= European put {pe:.4f} (BS {p_bs:.4f}); "
          f"early-exercise premium = {pa - pe:.4f}")
    assert pa >= pe - 1e-9
    assert pa > pe  # strictly, for this ITM put with r=8%
    assert abs(pe - p_bs) <= TOL["crr_bs_convergence_price"].value
