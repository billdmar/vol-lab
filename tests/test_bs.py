"""Black-Scholes engine tests: golden values, parity, IV round-trip, Greeks sanity.

Golden targets are hand-computed / standard-reference Black-Scholes values (see the
inline derivations). Tolerances that gate cross-engine claims are pulled by NAME from
config.tolerances so this file never silently invents its own bounds.
"""

from __future__ import annotations

import math

import pytest

from config.tolerances import TOL
from src.bs import BS, BlackScholes
from src.interfaces import GreeksEngine, IVSolver, Pricer


def _call(**kw) -> float:
    return BS.price(option_type="C", **kw).price


def _put(**kw) -> float:
    return BS.price(option_type="P", **kw).price


# --------------------------------------------------------------- protocol wiring
def test_black_scholes_satisfies_protocols():
    assert isinstance(BS, Pricer)
    assert isinstance(BS, GreeksEngine)
    assert isinstance(BS, IVSolver)
    # A fresh instance is equally valid (stateless).
    assert isinstance(BlackScholes(), Pricer)


# ------------------------------------------------------------------- golden values
def test_golden_atm_zero_rate():
    # S=K=100, tau=1, r=0, sigma=0.2. d1=0.1, d2=-0.1, price=100*(2N(0.1)-1)=7.96557...
    c = _call(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.2)
    p = _put(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.2)
    assert c == pytest.approx(7.965567455405804, abs=1e-4)
    # r=0 & F=K => put-call symmetry: call == put exactly.
    assert p == pytest.approx(c, abs=1e-4)


def test_golden_nonzero_rate_call():
    # S=K=100, tau=1, r=0.05, sigma=0.2 -> textbook BS call = 10.450583572...
    c = _call(spot=100.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.2)
    assert c == pytest.approx(10.450583572185565, abs=1e-4)


def test_golden_nonzero_rate_put_via_parity():
    # Same inputs; parity gives put = call - S + K*e^{-r} = 10.4506 - 100 + 95.1229 = 5.5735.
    p = _put(spot=100.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.2)
    assert p == pytest.approx(5.573526022256971, abs=1e-4)


def test_golden_with_carry_dividend():
    # S=100,K=95,tau=0.5,r=0.05,sigma=0.25,q=0.03. F=100*e^{0.01}=101.005.
    # call = df*(F*N(d1)-K*N(d2)) = 10.059924 (independently reproduced via scipy.stats.norm).
    c = _call(spot=100.0, strike=95.0, tau=0.5, rate=0.05, sigma=0.25, carry=0.03)
    assert c == pytest.approx(10.059923757343077, abs=1e-4)


# ---------------------------------------------------------------- degenerate limits
def test_tau_zero_returns_intrinsic():
    assert _call(spot=110.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.2) == pytest.approx(10.0)
    assert _put(spot=90.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.2) == pytest.approx(10.0)
    assert _call(spot=90.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.2) == pytest.approx(0.0)


def test_sigma_zero_returns_discounted_forward_payoff():
    # sigma=0: deterministic. call = df*max(F-K,0), F=spot*e^{(r-q)tau}.
    r, q, tau = 0.05, 0.0, 1.0
    fwd = 100.0 * math.exp((r - q) * tau)
    df = math.exp(-r * tau)
    expect = df * max(fwd - 100.0, 0.0)
    assert _call(spot=100.0, strike=100.0, tau=tau, rate=r, sigma=0.0) == pytest.approx(expect)
    # OTM deterministic call is worthless.
    assert _call(spot=100.0, strike=200.0, tau=tau, rate=r, sigma=0.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ put-call parity
def test_put_call_parity_machine_precision():
    tol = TOL["parity_model"].value
    grid = [
        (spot, strike, tau, rate, sigma, carry)
        for spot in (50.0, 100.0, 150.0)
        for strike in (80.0, 100.0, 120.0)
        for tau in (0.1, 1.0, 2.0)
        for rate in (0.0, 0.05)
        for sigma in (0.1, 0.4, 0.8)
        for carry in (0.0, 0.03)
    ]
    for spot, strike, tau, rate, sigma, carry in grid:
        c = _call(spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
        p = _put(spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
        df = math.exp(-rate * tau)
        eqt = math.exp(-carry * tau)
        # C - P = S*e^{-q*tau} - K*e^{-r*tau}
        lhs = c - p
        rhs = spot * eqt - strike * df
        assert abs(lhs - rhs) < tol, (spot, strike, tau, rate, sigma, carry, lhs - rhs)


# --------------------------------------------------------------------- IV round-trip
def test_iv_roundtrip_recovers_sigma():
    tol = TOL["iv_roundtrip_abs"].value
    cases = [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "C"),
        (100.0, 120.0, 0.5, 0.03, 0.35, 0.02, "C"),
        (100.0, 80.0, 2.0, 0.0, 0.65, 0.0, "P"),
        (60000.0, 55000.0, 0.25, 0.0, 0.75, 0.0, "P"),  # crypto-scale
        (100.0, 100.0, 0.05, 0.01, 0.90, 0.0, "C"),      # short-dated high vol
    ]
    for spot, strike, tau, rate, sigma0, carry, ot in cases:
        px = BS.price(
            spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma0,
            option_type=ot, carry=carry,
        ).price
        iv = BS.implied_vol(
            price=px, spot=spot, strike=strike, tau=tau, rate=rate,
            option_type=ot, carry=carry,
        )
        assert iv is not None
        assert iv == pytest.approx(sigma0, abs=tol), (ot, sigma0, iv)


def test_iv_roundtrip_dense_grid():
    # A denser sweep to be sure the solver is not fragile across the smile.
    tol = TOL["iv_roundtrip_abs"].value
    for strike in (70.0, 90.0, 100.0, 110.0, 130.0):
        for sigma0 in (0.15, 0.30, 0.55, 0.85):
            px = BS.price(
                spot=100.0, strike=strike, tau=0.75, rate=0.04, sigma=sigma0,
                option_type="C", carry=0.01,
            ).price
            iv = BS.implied_vol(
                price=px, spot=100.0, strike=strike, tau=0.75, rate=0.04,
                option_type="C", carry=0.01,
            )
            assert iv is not None
            assert iv == pytest.approx(sigma0, abs=tol)


def test_iv_matches_price_to_1e8():
    # The documented convergence claim: recovered IV reprices to ~1e-8.
    px = BS.price(
        spot=100.0, strike=105.0, tau=1.0, rate=0.05, sigma=0.42, option_type="C"
    ).price
    iv = BS.implied_vol(
        price=px, spot=100.0, strike=105.0, tau=1.0, rate=0.05, option_type="C"
    )
    assert iv is not None
    repriced = BS.price(
        spot=100.0, strike=105.0, tau=1.0, rate=0.05, sigma=iv, option_type="C"
    ).price
    assert abs(repriced - px) < 1e-8


# ------------------------------------------------------------ IV failure modes (honest None)
def test_iv_none_on_subintrinsic_price():
    # Call worth less than discounted intrinsic -> no arbitrage-free IV.
    intrinsic = math.exp(-0.05) * (100.0 * math.exp(0.05) - 90.0)  # = spot - K*df region
    below = intrinsic - 1.0
    iv = BS.implied_vol(
        price=below, spot=100.0, strike=90.0, tau=1.0, rate=0.05, option_type="C"
    )
    assert iv is None


def test_iv_none_at_or_above_ceiling():
    # Call price >= spot*e^{-q*tau} is the sigma->inf ceiling: no finite IV.
    iv = BS.implied_vol(
        price=100.0, spot=100.0, strike=100.0, tau=1.0, rate=0.05, option_type="C"
    )
    assert iv is None


def test_iv_none_on_expired():
    iv = BS.implied_vol(
        price=5.0, spot=100.0, strike=95.0, tau=0.0, rate=0.05, option_type="C"
    )
    assert iv is None


def test_iv_none_deep_otm_near_expiry_vega_collapse():
    # Deep-OTM, near-expiry: BS price underflows to ~0 (== intrinsic). Vega collapses,
    # so no vol is recoverable -> honest None rather than a meaningless ~0.
    px = BS.price(
        spot=100.0, strike=300.0, tau=0.002, rate=0.0, sigma=0.5, option_type="C"
    ).price
    assert px < 1e-12  # confirm the underflow precondition
    iv = BS.implied_vol(
        price=px, spot=100.0, strike=300.0, tau=0.002, rate=0.0, option_type="C"
    )
    assert iv is None


# ------------------------------------------------------- Greeks degenerate limits
def test_greeks_sigma_zero_call_itm_and_otm():
    # sigma=0 => step-function delta, zero gamma/vega. ITM call (F>K) has delta e^{-q*tau}.
    g_itm = BS.greeks(spot=120.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.0,
                      option_type="C", carry=0.02)
    assert g_itm.gamma == 0.0 and g_itm.vega == 0.0
    assert g_itm.delta == pytest.approx(math.exp(-0.02))
    assert g_itm.rho > 0.0  # ITM call rho = K*tau*df
    g_otm = BS.greeks(spot=80.0, strike=200.0, tau=1.0, rate=0.05, sigma=0.0,
                      option_type="C", carry=0.02)
    assert g_otm.delta == 0.0 and g_otm.rho == 0.0 and g_otm.theta == 0.0


def test_greeks_sigma_zero_put_itm_and_otm():
    # ITM put (F<K): delta -e^{-q*tau}, rho < 0.
    g_itm = BS.greeks(spot=80.0, strike=120.0, tau=1.0, rate=0.05, sigma=0.0,
                      option_type="P", carry=0.02)
    assert g_itm.gamma == 0.0 and g_itm.vega == 0.0
    assert g_itm.delta == pytest.approx(-math.exp(-0.02))
    assert g_itm.rho < 0.0
    g_otm = BS.greeks(spot=200.0, strike=80.0, tau=1.0, rate=0.05, sigma=0.0,
                      option_type="P", carry=0.02)
    assert g_otm.delta == 0.0 and g_otm.rho == 0.0


def test_greeks_tau_zero_limit():
    # Expired ITM call: delta ~ e^{0}=1 (tau=0 => eqt=df=1), gamma/vega 0.
    g = BS.greeks(spot=120.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.2, option_type="C")
    assert g.gamma == 0.0 and g.vega == 0.0
    assert g.delta == pytest.approx(1.0)


def test_iv_roundtrip_very_high_vol():
    # sigma0 > 1 forces the solver's bracket to grow past hi=1.0 before straddling.
    sigma0 = 2.5
    px = BS.price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=sigma0,
                  option_type="C").price
    iv = BS.implied_vol(price=px, spot=100.0, strike=100.0, tau=1.0, rate=0.0,
                        option_type="C")
    assert iv is not None
    assert iv == pytest.approx(sigma0, abs=TOL["iv_roundtrip_abs"].value)


def test_iv_solver_vega_floor_and_bracket_escape():
    # Deep-OTM, very short tau, high vol: vega goes below the floor near the root and
    # Newton steps escape the bracket, so the solver falls back to bisection. It still
    # converges the PRICE to ~1e-8; the recovered VOL is looser here (the documented
    # deep-OTM/near-expiry regime, > iv_roundtrip_abs), so we assert on price, not vol.
    sigma0 = 1.0
    px = BS.price(spot=100.0, strike=200.0, tau=0.02, rate=0.0, sigma=sigma0,
                  option_type="C").price
    assert px > 1e-12  # above intrinsic: a (looser) IV is still recoverable
    iv = BS.implied_vol(price=px, spot=100.0, strike=200.0, tau=0.02, rate=0.0,
                        option_type="C")
    assert iv is not None
    repriced = BS.price(spot=100.0, strike=200.0, tau=0.02, rate=0.0, sigma=iv,
                        option_type="C").price
    assert abs(repriced - px) < 1e-6
    assert iv == pytest.approx(sigma0, abs=1e-2)  # loose: deep-OTM vol resolution


def test_iv_roundtrip_deep_otm_high_vol():
    # Deep-OTM but with enough vol/time that a real (large) IV exists: exercises the
    # bracket growth and safeguarded stepping without collapsing to intrinsic.
    sigma0 = 1.8
    px = BS.price(spot=100.0, strike=250.0, tau=1.5, rate=0.0, sigma=sigma0,
                  option_type="C").price
    iv = BS.implied_vol(price=px, spot=100.0, strike=250.0, tau=1.5, rate=0.0,
                        option_type="C")
    assert iv is not None
    assert iv == pytest.approx(sigma0, abs=TOL["iv_roundtrip_abs"].value)


# --------------------------------------------------------------------- Greeks sanity
def test_greeks_signs_and_bounds_call():
    g = BS.greeks(spot=100.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.2, option_type="C")
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert 0.0 < g.delta < 1.0
    assert g.theta < 0.0   # long call decays in calendar time
    assert g.rho > 0.0     # call value rises with rate


def test_greeks_signs_and_bounds_put():
    g = BS.greeks(spot=100.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.2, option_type="P")
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert -1.0 < g.delta < 0.0
    assert g.rho < 0.0     # put value falls with rate


def test_gamma_vega_type_independent():
    # gamma and vega do not depend on option_type.
    gc = BS.greeks(spot=100.0, strike=110.0, tau=0.5, rate=0.03, sigma=0.3, option_type="C")
    gp = BS.greeks(spot=100.0, strike=110.0, tau=0.5, rate=0.03, sigma=0.3, option_type="P")
    assert gc.gamma == pytest.approx(gp.gamma, rel=1e-14)
    assert gc.vega == pytest.approx(gp.vega, rel=1e-14)


def test_delta_matches_finite_difference():
    # Sanity that the closed-form delta agrees with a bump (cross-checked in G1 too).
    kw = {"spot": 100.0, "strike": 105.0, "tau": 1.0, "rate": 0.05, "sigma": 0.25, "carry": 0.02}
    h = 1e-4
    up = BS.price(spot=kw["spot"] + h, strike=kw["strike"], tau=kw["tau"],
                  rate=kw["rate"], sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    dn = BS.price(spot=kw["spot"] - h, strike=kw["strike"], tau=kw["tau"],
                  rate=kw["rate"], sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    fd_delta = (up - dn) / (2 * h)
    g = BS.greeks(option_type="C", **kw)
    assert g.delta == pytest.approx(fd_delta, rel=1e-6)


def test_vega_matches_finite_difference():
    kw = {"spot": 100.0, "strike": 105.0, "tau": 1.0, "rate": 0.05, "sigma": 0.25, "carry": 0.02}
    h = 1e-5
    up = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"], rate=kw["rate"],
                  sigma=kw["sigma"] + h, option_type="C", carry=kw["carry"]).price
    dn = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"], rate=kw["rate"],
                  sigma=kw["sigma"] - h, option_type="C", carry=kw["carry"]).price
    fd_vega = (up - dn) / (2 * h)
    g = BS.greeks(option_type="C", **kw)
    assert g.vega == pytest.approx(fd_vega, rel=1e-5)


def test_theta_matches_finite_difference():
    # theta = dP/dt (per year); price is decreasing in tau for a long option, and
    # dP/dt = -dP/dtau, so bump tau and negate.
    kw = {"spot": 100.0, "strike": 100.0, "tau": 1.0, "rate": 0.05, "sigma": 0.25, "carry": 0.0}
    h = 1e-5
    up = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"] + h, rate=kw["rate"],
                  sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    dn = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"] - h, rate=kw["rate"],
                  sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    fd_theta = -(up - dn) / (2 * h)
    g = BS.greeks(option_type="C", **kw)
    assert g.theta == pytest.approx(fd_theta, rel=1e-5)


def test_rho_matches_finite_difference():
    kw = {"spot": 100.0, "strike": 100.0, "tau": 1.0, "rate": 0.05, "sigma": 0.25, "carry": 0.0}
    h = 1e-6
    up = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"], rate=kw["rate"] + h,
                  sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    dn = BS.price(spot=kw["spot"], strike=kw["strike"], tau=kw["tau"], rate=kw["rate"] - h,
                  sigma=kw["sigma"], option_type="C", carry=kw["carry"]).price
    fd_rho = (up - dn) / (2 * h)
    g = BS.greeks(option_type="C", **kw)
    assert g.rho == pytest.approx(fd_rho, rel=1e-5)


# ------------------------------------------------------------------- input validation
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BS.price(spot=-1.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.2, option_type="C")
    with pytest.raises(ValueError):
        BS.price(spot=100.0, strike=0.0, tau=1.0, rate=0.0, sigma=0.2, option_type="C")
    with pytest.raises(ValueError):
        BS.price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=-0.2, option_type="C")
    with pytest.raises(ValueError):
        BS.price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.2, option_type="X")
