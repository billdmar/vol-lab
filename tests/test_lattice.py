"""Tests for the CRR binomial-tree engine (src/lattice).

Two engines verify each other here: the CRR tree (SA-lattice) and a closed-form
Black-Scholes reference defined *locally* in this file. The local reference exists
because src.bs is populated in parallel by another subagent; a self-contained
Gaussian-CDF BS is an entirely independent algorithm from a recombining lattice, so
using it as the convergence target is genuine cross-verification, not a tautology.
The convergence harness accepts an injected `bs_pricer`, which we use here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from config.tolerances import TOL
from src.interfaces import Pricer, PriceResult
from src.lattice import CRRBinomial, convergence_order
from src.lattice.crr import _crr_price
from src.schema import OptionType

# ------------------------------------------------------------------ BS reference


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency in this test)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class _RefBS:
    """Independent closed-form Black-Scholes pricer (frozen conventions).

    Forward F = spot*exp((rate-carry)*tau); discount df = exp(-rate*tau). This is the
    reference the CRR tree must converge to. Kept minimal and self-contained so this
    test never depends on the in-flight src.bs module.
    """

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
        fwd = spot * math.exp((rate - carry) * tau)
        df = math.exp(-rate * tau)
        if tau <= 0.0 or sigma <= 0.0:
            payoff = max(fwd - strike, 0.0) if option_type == "C" else max(strike - fwd, 0.0)
            return PriceResult(price=df * payoff)
        vol = sigma * math.sqrt(tau)
        d1 = (math.log(fwd / strike) + 0.5 * vol * vol) / vol
        d2 = d1 - vol
        if option_type == "C":
            val = df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
        elif option_type == "P":
            val = df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))
        else:
            raise ValueError(f"bad option_type {option_type!r}")
        return PriceResult(price=val)


REF = _RefBS()

# A base market for European convergence: unit spot, one-year, moderate vol, no rates
# (the coin-margined Deribit default r=q=0). Strikes span ITM/ATM/OTM.
BASE = {"spot": 100.0, "tau": 1.0, "rate": 0.0, "sigma": 0.35, "carry": 0.0}
STRIKES = (80.0, 100.0, 130.0)


# --------------------------------------------------------------- sanity / protocol


def test_crr_implements_pricer_protocol():
    assert isinstance(CRRBinomial(), Pricer)


def test_tau_zero_returns_intrinsic():
    # At expiry the tree must return exactly the payoff.
    c = _crr_price(spot=120.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.3,
                   option_type="C", carry=0.0, steps=10, american=False)
    p = _crr_price(spot=90.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.3,
                   option_type="P", carry=0.0, steps=10, american=False)
    assert c == pytest.approx(20.0, abs=1e-12)
    assert p == pytest.approx(10.0, abs=1e-12)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        _crr_price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.0,
                   option_type="C", carry=0.0, steps=100, american=False)
    with pytest.raises(ValueError):
        _crr_price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.3,
                   option_type="C", carry=0.0, steps=0, american=False)
    with pytest.raises(ValueError):
        _crr_price(spot=100.0, strike=100.0, tau=1.0, rate=0.0, sigma=0.3,
                   option_type="X", carry=0.0, steps=100, american=False)  # type: ignore[arg-type]


def test_arbitrageable_tree_raises():
    """A valid sigma but a drift that escapes [d, u] makes the risk-neutral p leave [0,1];
    the tree is arbitrageable and the engine must FAIL LOUD, not clip p into a fake value.

    Here exp((rate-carry)*dt) > u = exp(sigma*sqrt(dt)) because |rate-carry| is large vs a
    small sigma at coarse dt, so p = (growth - d)/(u - d) > 1. (Distinct from sigma<=0,
    which trips the earlier guard.)
    """
    with pytest.raises(ValueError, match="risk-neutral p"):
        _crr_price(spot=100.0, strike=100.0, tau=1.0, rate=0.9, sigma=0.05,
                   option_type="C", carry=0.0, steps=50, american=False)


# ------------------------------------------------------- European -> BS at N=2000


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("otype", ["C", "P"])
def test_european_converges_to_bs_at_2000(strike: float, otype: OptionType):
    tol = TOL["crr_bs_convergence_price"].value
    crr = CRRBinomial(steps=2000, american=False)
    got = crr.price(strike=strike, option_type=otype, **BASE).price
    ref = REF.price(strike=strike, option_type=otype, **BASE).price
    assert abs(got - ref) <= tol, f"strike={strike} {otype}: |{got}-{ref}|={abs(got-ref):.2e}"


def test_european_converges_with_nonzero_rate_and_carry():
    # Exercise the full (rate, carry) drift path against BS.
    params = {"spot": 100.0, "strike": 105.0, "tau": 0.75, "rate": 0.05,
              "sigma": 0.4, "carry": 0.02}
    tol = TOL["crr_bs_convergence_price"].value
    for otype in ("C", "P"):
        got = CRRBinomial(steps=2000).price(option_type=otype, **params).price
        ref = REF.price(option_type=otype, **params).price
        assert abs(got - ref) <= tol


# --------------------------------------------------------- measured convergence order


def test_convergence_order_measured_and_reported(capsys):
    tol_order = TOL["crr_convergence_order_min"].value
    res = convergence_order(strike=100.0, option_type="C", bs_pricer=REF, **BASE)
    order = res["order"]
    # Report the measured value (visible with pytest -s), per the mission's
    # "measured, not just the pass" requirement.
    errs_fmt = [f"{e:.2e}" for e in res["errors"]]
    print(f"\n[CRR] measured convergence order (ATM call) = {order:.4f} "
          f"(min required {tol_order}); errors={errs_fmt}")
    assert order >= tol_order, f"measured order {order:.4f} < required {tol_order}"
    # CRR is first-order; a correct tree sits comfortably below ~1.5.
    assert order <= 1.6, f"order {order:.4f} implausibly high for first-order CRR"


def test_convergence_order_for_a_put():
    tol_order = TOL["crr_convergence_order_min"].value
    res = convergence_order(strike=110.0, option_type="P", bs_pricer=REF, **BASE)
    assert res["order"] >= tol_order


def test_convergence_errors_decrease_overall():
    # The error at the largest N must be smaller than at the smallest N.
    res = convergence_order(strike=100.0, option_type="C", bs_pricer=REF, **BASE)
    errs = res["errors"]
    assert errs[-1] < errs[0]


def test_convergence_order_needs_two_points():
    with pytest.raises(ValueError):
        convergence_order(strike=100.0, option_type="C", bs_pricer=REF,
                          steps=(100,), **BASE)


# ----------------------------------------------- American >= European (early ex.)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("otype", ["C", "P"])
def test_american_ge_european(strike: float, otype: OptionType):
    # Early-exercise premium is non-negative for identical params (use a rate/carry
    # regime where early exercise can matter).
    params = {"spot": 100.0, "strike": strike, "tau": 1.0, "rate": 0.06,
              "sigma": 0.35, "carry": 0.03}
    euro = CRRBinomial(steps=1000, american=False).price(option_type=otype, **params).price
    amer = CRRBinomial(steps=1000, american=True).price(option_type=otype, **params).price
    assert amer >= euro - 1e-10, f"American {amer} < European {euro}"


def test_itm_american_put_strictly_greater():
    # A deep ITM put with a positive rate makes early exercise optimal (recover the
    # strike now, earn interest) -> strictly positive early-exercise premium.
    params = {"spot": 70.0, "strike": 100.0, "tau": 1.0, "rate": 0.10,
              "sigma": 0.30, "carry": 0.0}
    euro = CRRBinomial(steps=1500, american=False).price(option_type="P", **params).price
    amer = CRRBinomial(steps=1500, american=True).price(option_type="P", **params).price
    assert amer > euro + 1e-3, f"expected early-exercise premium, euro={euro}, amer={amer}"


def test_american_call_no_dividend_equals_european():
    # Classic result: an American call on a non-dividend-paying (carry=0, rate>0)
    # underlying is never exercised early -> equals the European call.
    params = {"spot": 100.0, "strike": 95.0, "tau": 1.0, "rate": 0.08,
              "sigma": 0.3, "carry": 0.0}
    euro = CRRBinomial(steps=1500, american=False).price(option_type="C", **params).price
    amer = CRRBinomial(steps=1500, american=True).price(option_type="C", **params).price
    assert amer == pytest.approx(euro, abs=1e-6)


# ------------------------------------------------------- put-call parity on the tree


@pytest.mark.parametrize("strike", STRIKES)
def test_european_put_call_parity_on_tree(strike: float):
    # C - P = S*exp(-q*tau) - K*exp(-r*tau) should hold on European CRR prices to a
    # loose lattice tolerance (the tree discretization error, not machine precision).
    params = {"spot": 100.0, "strike": strike, "tau": 1.0, "rate": 0.05,
              "sigma": 0.35, "carry": 0.02}
    crr = CRRBinomial(steps=2000, american=False)
    call = crr.price(option_type="C", **params).price
    put = crr.price(option_type="P", **params).price
    lhs = call - put
    rhs = (params["spot"] * math.exp(-params["carry"] * params["tau"])
           - strike * math.exp(-params["rate"] * params["tau"]))
    # Loose lattice band: parity is exact on the tree in exact arithmetic, so this
    # only absorbs float rounding across the two full backward inductions.
    assert abs(lhs - rhs) <= 1e-3, f"parity resid {abs(lhs - rhs):.2e} at K={strike}"


# ----------------------------------------------------------------- determinism


def test_deterministic_repeat():
    crr = CRRBinomial(steps=500)
    a = crr.price(strike=100.0, option_type="C", **BASE).price
    b = crr.price(strike=100.0, option_type="C", **BASE).price
    assert a == b
