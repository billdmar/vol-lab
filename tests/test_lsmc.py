"""LSMC American pricer verification.

The reference for American vanilla is a fine CRR lattice (steps=2000). We assert
the Longstaff-Schwartz price lands within the registered relative tolerance on a
set of ITM/ATM American puts (where early exercise carries real value at a 5-8%
rate), that the American price never sits below the European Black-Scholes price
(early-exercise premium >= 0), that a no-early-exercise American call reduces to
the European price within the MC CI, and that a fixed seed is bit-for-bit
reproducible.

Measured LSMC-vs-CRR relative differences and the LSMC CIs are printed by the
report test so the headline claim is backed by fresh evidence, not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.tolerances import TOL
from src.bs import BS
from src.lattice.crr import CRRBinomial
from src.lsmc import LSMCPricer

# Paths/steps chosen so the LSMC-vs-CRR band comfortably clears the registered
# 1% tolerance at seed 12345 while keeping the suite fast (~a few seconds).
N_PATHS = 100_000
N_STEPS = 50
SEED = 12345

# ITM / ATM American puts at 5-8% rates: the regime where the early-exercise
# right is worth the most, so a broken continuation regression fails loudly.
AMERICAN_PUT_CASES = [
    {"spot": 100.0, "strike": 100.0, "tau": 1.0, "rate": 0.06, "sigma": 0.30, "option_type": "P"},
    {"spot": 90.0, "strike": 100.0, "tau": 1.0, "rate": 0.06, "sigma": 0.30, "option_type": "P"},
    {"spot": 100.0, "strike": 110.0, "tau": 0.5, "rate": 0.08, "sigma": 0.40, "option_type": "P"},
    {"spot": 95.0, "strike": 100.0, "tau": 0.75, "rate": 0.05, "sigma": 0.25, "option_type": "P"},
]

CRR_AMERICAN = CRRBinomial(steps=2000, american=True)


def _lsmc(**kw):
    return LSMCPricer(n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED).price(**kw)


@pytest.mark.parametrize("case", AMERICAN_PUT_CASES)
def test_lsmc_vs_crr_american_put(case):
    """LSMC American put within TOL['lsmc_vs_crr_american_rel'] of a fine CRR lattice."""
    tol = TOL["lsmc_vs_crr_american_rel"]
    assert tol.kind == "rel"
    lsmc = _lsmc(**case)
    ref = CRR_AMERICAN.price(**case).price
    rel = abs(lsmc.price - ref) / ref
    assert rel <= tol.value, (
        f"LSMC={lsmc.price:.5f} vs CRR={ref:.5f} rel={rel:.4%} exceeds {tol.value:.2%}"
    )


@pytest.mark.parametrize("case", AMERICAN_PUT_CASES)
def test_american_ge_european(case):
    """American value >= European BS value: the early-exercise premium is non-negative.

    Allow a hair of MC slack (one stderr) so pure sampling noise on a case with a
    near-zero premium does not flip the inequality into a false failure.
    """
    lsmc = _lsmc(**case)
    euro = BS.price(**case).price
    slack = lsmc.stderr or 0.0
    assert lsmc.price >= euro - slack, (
        f"American LSMC={lsmc.price:.5f} below European BS={euro:.5f} "
        f"(stderr={slack:.5f})"
    )


def test_american_call_no_dividend_matches_european():
    """American call with carry=0 is never exercised early -> equals European in CI.

    With no dividend/carry the continuation value of a call always dominates its
    intrinsic, so the American right is worthless and the price collapses to the
    European one; we require the European price to lie inside the LSMC 95% CI.
    """
    case = {
        "spot": 100.0, "strike": 95.0, "tau": 1.0, "rate": 0.05, "sigma": 0.30,
        "option_type": "C", "carry": 0.0,
    }
    lsmc = _lsmc(**case)
    euro = BS.price(**case).price
    lo, hi = lsmc.ci95
    assert lo <= euro <= hi, (
        f"European BS={euro:.5f} outside LSMC CI=({lo:.5f}, {hi:.5f})"
    )


def test_determinism_same_seed_identical():
    """Same seed -> bit-for-bit identical price (mandatory reproducibility)."""
    case = AMERICAN_PUT_CASES[0]
    a = LSMCPricer(n_paths=50_000, n_steps=40, seed=777).price(**case)
    b = LSMCPricer(n_paths=50_000, n_steps=40, seed=777).price(**case)
    assert a.price == b.price
    assert a.stderr == b.stderr
    assert a.ci95 == b.ci95


def test_different_seed_changes_price():
    """A different seed perturbs the estimate (sanity: the seed is actually used)."""
    case = AMERICAN_PUT_CASES[0]
    a = LSMCPricer(n_paths=50_000, n_steps=40, seed=1).price(**case)
    b = LSMCPricer(n_paths=50_000, n_steps=40, seed=2).price(**case)
    assert a.price != b.price


def test_tau_zero_returns_intrinsic():
    """tau <= 0 -> immediate intrinsic, zero stderr (no randomness)."""
    itm = LSMCPricer().price(
        spot=90.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.30, option_type="P"
    )
    assert itm.price == pytest.approx(10.0)
    assert itm.stderr == 0.0
    assert itm.ci95 == (10.0, 10.0)

    otm = LSMCPricer().price(
        spot=110.0, strike=100.0, tau=0.0, rate=0.05, sigma=0.30, option_type="P"
    )
    assert otm.price == 0.0


def test_sigma_zero_deterministic_american():
    """sigma <= 0 -> deterministic forward trajectory; exercise the best discounted date.

    Here the put is ITM at inception (S=100 < K=110) and the riskless forward only
    drifts further from the strike, so the optimal policy exercises immediately for
    the full intrinsic of 10; stderr is zero (no randomness).
    """
    res = LSMCPricer(n_steps=50).price(
        spot=100.0, strike=110.0, tau=1.0, rate=0.05, sigma=0.0, option_type="P"
    )
    assert res.price == pytest.approx(10.0)
    assert res.stderr == 0.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        LSMCPricer().price(
            spot=100.0, strike=100.0, tau=1.0, rate=0.05, sigma=0.3, option_type="X"
        )


def test_constructor_validation():
    with pytest.raises(ValueError, match="n_paths"):
        LSMCPricer(n_paths=1)
    with pytest.raises(ValueError, match="n_steps"):
        LSMCPricer(n_steps=0)
    with pytest.raises(ValueError, match="basis_degree"):
        LSMCPricer(basis_degree=0)


def test_report_lsmc_vs_crr(capsys):
    """Print the measured LSMC-vs-CRR relative diffs + LSMC CIs (evidence for headline)."""
    rels = []
    with capsys.disabled():
        print("\n  LSMC American put vs CRR(2000) lattice:")
        for case in AMERICAN_PUT_CASES:
            lsmc = _lsmc(**case)
            ref = CRR_AMERICAN.price(**case).price
            euro = BS.price(**case).price
            rel = abs(lsmc.price - ref) / ref
            rels.append(rel)
            lo, hi = lsmc.ci95
            print(
                f"    S={case['spot']:.0f} K={case['strike']:.0f} r={case['rate']:.0%}: "
                f"LSMC={lsmc.price:.4f} CI=({lo:.4f},{hi:.4f}) "
                f"CRR={ref:.4f} rel={rel:.4%} euro={euro:.4f}"
            )
        print(f"    max |LSMC-CRR|/CRR = {max(rels):.4%} "
              f"(tol {TOL['lsmc_vs_crr_american_rel'].value:.2%})")
    assert max(rels) <= TOL["lsmc_vs_crr_american_rel"].value
    assert np.isfinite(max(rels))
