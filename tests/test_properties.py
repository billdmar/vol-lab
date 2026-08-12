"""Cross-engine property-based tests (hypothesis).

These assert *invariants of the mathematics* that must hold for arbitrary valid
inputs, not hand-picked golden values — the complement to the golden-value suites
in test_bs.py / test_lattice.py. The primary engine under test is the closed-form
Black-Scholes (`src.bs.BS`); the American>=European property additionally uses the
CRR lattice (`src.lattice.CRRBinomial`), since early exercise is a lattice feature.

Determinism (project hard rule): a fixed hypothesis profile with `derandomize=True`
seeds example generation from the test identity, so every run explores the SAME
inputs. `database=None` disables the cross-run example cache, and `deadline=None`
removes wall-clock flakiness (arm64 dev vs ubuntu CI differ in speed). Tolerances
are pulled BY NAME from config.tolerances so this file never invents its own bounds.

Properties covered:
  1. Price bounds     — discounted-forward intrinsic <= price <= no-arb ceiling.
  2. Monotonicity     — price up in sigma (vega>0); call up / put down in spot;
                        call down / put up in strike.
  3. Put-call parity  — C - P == S*e^{-q*tau} - K*e^{-r*tau} to TOL["parity_model"].
  4. American >= European (CRR) — early-exercise right is never worth negative.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from config.tolerances import TOL
from src.bs import BS
from src.lattice import CRRBinomial

# --------------------------------------------------------------------- profile
# Deterministic + bounded runtime. derandomize makes the generated examples a pure
# function of the test, so CI and local agree exactly and reruns never "flake in".
settings.register_profile(
    "vol_lab",
    max_examples=300,
    deadline=None,
    database=None,
    derandomize=True,
)
settings.load_profile("vol_lab")


# ------------------------------------------------------------------ strategies
# Realistic-but-safe ranges. Spot/strike are kept <= 5000 for the parity property
# so its ABSOLUTE 1e-10 bound stays honest (parity float error scales with price
# magnitude: ~3e-12 at spot<=5000, but ~5e-11 near crypto-scale 100k). Sigma spans
# low-vol equities to high-vol crypto; rate/carry cover the modest ranges we use.
_spot = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
_strike = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
_tau = st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False)
_rate = st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
_sigma = st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)
_carry = st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)


def _price(option_type: str, **kw) -> float:
    return BS.price(option_type=option_type, **kw).price


# ------------------------------------------------------------ 1. price bounds
@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_call_price_bounds(spot, strike, tau, rate, sigma, carry):
    """Discounted-forward intrinsic <= call <= spot*e^{-q*tau} (the sigma->inf ceiling).

    We use the *discounted-forward* intrinsic df*max(F-K,0), not the naive max(S-K,0):
    under nonzero rate/carry the naive form is a strictly wrong lower bound (a deep-ITM
    European call can trade below S-K by the cost of carry). The upper bound is the
    zero-strike / infinite-vol limit. A tiny abs slack absorbs float64 rounding.
    """
    c = _price("C", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    df = math.exp(-rate * tau)
    eqt = math.exp(-carry * tau)
    fwd = spot * math.exp((rate - carry) * tau)
    lower = df * max(fwd - strike, 0.0)
    upper = spot * eqt
    slack = 1e-7 * max(1.0, spot)
    assert c >= lower - slack, (c, lower)
    assert c <= upper + slack, (c, upper)


@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_put_price_bounds(spot, strike, tau, rate, sigma, carry):
    """Discounted-forward intrinsic <= put <= K*e^{-r*tau} (the sigma->inf ceiling)."""
    p = _price("P", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    df = math.exp(-rate * tau)
    fwd = spot * math.exp((rate - carry) * tau)
    lower = df * max(strike - fwd, 0.0)
    upper = strike * df
    slack = 1e-7 * max(1.0, strike)
    assert p >= lower - slack, (p, lower)
    assert p <= upper + slack, (p, upper)


# ----------------------------------------------------------- 2. monotonicity
# A shared float-noise slack: prices here are <= ~5000 so absolute rounding is
# ~price*eps ~ 1e-12; 1e-8 catches a real sign flip while ignoring rounding.
_MONO_SLACK = 1e-8


@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_price_increases_in_sigma(spot, strike, tau, rate, sigma, carry):
    """Vega > 0: both calls and puts are non-decreasing in volatility."""
    assume(sigma <= 1.99)  # leave room for the +bump inside the strategy ceiling
    dv = 1e-3
    for ot in ("C", "P"):
        lo = _price(ot, spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
        hi = _price(ot, spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma + dv, carry=carry)
        assert hi >= lo - _MONO_SLACK, (ot, sigma, lo, hi)


@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_call_up_put_down_in_spot(spot, strike, tau, rate, sigma, carry):
    """Call delta in (0,1) -> up in spot; put delta in (-1,0) -> down in spot."""
    ds = 1e-3 * spot
    c_lo = _price("C", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    c_hi = _price("C", spot=spot + ds, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p_lo = _price("P", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p_hi = _price("P", spot=spot + ds, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    assert c_hi >= c_lo - _MONO_SLACK, (c_lo, c_hi)
    assert p_hi <= p_lo + _MONO_SLACK, (p_lo, p_hi)


@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_call_down_put_up_in_strike(spot, strike, tau, rate, sigma, carry):
    """Call decreases in strike; put increases in strike (dPrice/dK signs)."""
    dk = 1e-3 * strike
    c_lo = _price("C", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    c_hi = _price("C", spot=spot, strike=strike + dk, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p_lo = _price("P", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p_hi = _price("P", spot=spot, strike=strike + dk, tau=tau, rate=rate, sigma=sigma, carry=carry)
    assert c_hi <= c_lo + _MONO_SLACK, (c_lo, c_hi)
    assert p_hi >= p_lo - _MONO_SLACK, (p_lo, p_hi)


# ------------------------------------------------------------- 3. put-call parity
@given(
    spot=_spot, strike=_strike, tau=_tau, rate=_rate, sigma=_sigma, carry=_carry
)
def test_put_call_parity(spot, strike, tau, rate, sigma, carry):
    """C - P == S*e^{-q*tau} - K*e^{-r*tau} to machine precision (TOL['parity_model']).

    Parity is an algebraic identity of the closed form, so the only error is float64
    rounding over a handful of exp/mul ops. The bound is ABSOLUTE; spot/strike are
    capped at 5000 (see strategies) so the magnitude-scaled rounding stays under 1e-10.
    """
    tol = TOL["parity_model"].value
    c = _price("C", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p = _price("P", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    lhs = c - p
    rhs = spot * math.exp(-carry * tau) - strike * math.exp(-rate * tau)
    assert abs(lhs - rhs) < tol, (spot, strike, tau, rate, sigma, carry, lhs - rhs)


# ---------------------------------------------------- 4. American >= European (CRR)
# Same tree size for both legs so the comparison is exact (the early-exercise premium
# is >= 0 node-by-node on one lattice). Moderate spot/strike/steps keep it fast.
_am_spot = st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_am_strike = st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_am_tau = st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False)
_am_sigma = st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False)

_CRR_STEPS = 256
_CRR_EUR = CRRBinomial(steps=_CRR_STEPS, american=False)
_CRR_AME = CRRBinomial(steps=_CRR_STEPS, american=True)


@given(
    spot=_am_spot,
    strike=_am_strike,
    tau=_am_tau,
    rate=_rate,
    sigma=_am_sigma,
    carry=_carry,
)
@settings(max_examples=150)  # CRR at 256 steps is heavier than a closed-form eval
def test_american_ge_european(spot, strike, tau, rate, sigma, carry):
    """American price >= European price for both calls and puts on the SAME CRR tree.

    The American recursion takes max(continuation, intrinsic) at every node, so it can
    only add value. We `assume` the tree is a valid recombining lattice (risk-neutral
    p in [0,1]); if |rate-carry|*sqrt(dt) exceeds sigma the CRR moves cannot span the
    drift and the engine correctly raises — those inputs are out of the tree's regime,
    not a property violation.
    """
    dt = tau / _CRR_STEPS
    # CRR validity: growth exp((r-q)dt) must sit within [d, u] = [e^{-s√dt}, e^{s√dt}].
    assume(abs(rate - carry) * math.sqrt(dt) <= sigma)
    for ot in ("C", "P"):
        eur = _CRR_EUR.price(
            spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, option_type=ot, carry=carry
        ).price
        ame = _CRR_AME.price(
            spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, option_type=ot, carry=carry
        ).price
        assert ame >= eur - 1e-9, (ot, eur, ame)


# ------------------------------------------------------- 5. negative-rate regime
# Crypto/EUR options can trade under negative rates; nothing in the BS forward/discount
# or the parity identity assumes rate >= 0, but the other properties only sample rate in
# [0, 0.15]. This exercises the sign-negative regime explicitly across price bounds and
# parity, so a latent sign assumption in the forward/discount would surface here.
# Spot/strike/tau/sigma stay modest: parity is an ABSOLUTE-1e-10 identity whose float
# error scales with price*(1-e^{-q*tau}) and with N(d1) saturation at large sigma*sqrt(tau)
# -- the same magnitude discipline the rate>=0 parity property relies on. The negative
# rate (not the extreme-vol corner) is what this test is about.
_neg_rate = st.floats(min_value=-0.10, max_value=-1e-4, allow_nan=False, allow_infinity=False)
_nr_spot = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_nr_strike = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_nr_tau = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
_nr_sigma = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
_nr_carry = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)


@given(spot=_nr_spot, strike=_nr_strike, tau=_nr_tau, rate=_neg_rate,
       sigma=_nr_sigma, carry=_nr_carry)
def test_negative_rate_bounds_and_parity(spot, strike, tau, rate, sigma, carry):
    """Price bounds and put-call parity still hold under negative rates.

    Under r < 0 the discount df = e^{-r*tau} > 1, and the no-arb ceilings/floors follow
    the same discounted-forward formulas (they never assumed a sign). Parity is the same
    algebraic identity. If any of these used max(S-K,0) or an r>=0 shortcut, this fails.
    """
    c = _price("C", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    p = _price("P", spot=spot, strike=strike, tau=tau, rate=rate, sigma=sigma, carry=carry)
    df = math.exp(-rate * tau)
    eqt = math.exp(-carry * tau)
    fwd = spot * math.exp((rate - carry) * tau)
    slack = 1e-7 * max(1.0, spot, strike)
    # Bounds (discounted-forward intrinsic <= price <= sigma->inf ceiling).
    assert c >= df * max(fwd - strike, 0.0) - slack
    assert c <= spot * eqt + slack
    assert p >= df * max(strike - fwd, 0.0) - slack
    assert p <= strike * df + slack
    # Parity identity. Parity float error scales with price magnitude (and with N(d1)
    # saturation for deep/short options), so this property test — which spans a range of
    # magnitudes — uses a RELATIVE bound rather than the closed-form ABSOLUTE parity_model
    # tolerance (the latter is exercised at fixed <=5000 magnitude by test_put_call_parity).
    lhs = c - p
    rhs = spot * eqt - strike * df
    scale = max(abs(c), abs(p), spot * eqt, strike * df, 1.0)
    assert abs(lhs - rhs) <= 1e-10 * scale, (spot, strike, tau, rate, sigma, carry, lhs - rhs)
