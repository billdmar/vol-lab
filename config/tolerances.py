"""Tolerance registry (single source of truth for every differential tolerance).

ONE place for every differential tolerance in the test suite, each with a written
justification. The hard rule: a tolerance is NEVER widened to make a failing test
pass. A failure means investigate and write down the finding — the tolerance only
moves if the *justification* changes, and that is a deliberate decision recorded
here in the commit message.

Each entry is a `Tol` with:
  * value        — the numeric threshold used by the test
  * kind         — 'abs' | 'rel' | 'order' | 'prob' (how it's compared)
  * why          — the mathematical/statistical reason this bound is correct-and-tight

Import as:  from config.tolerances import TOL
and reference by name:  TOL["parity_model"].value
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tol:
    value: float
    kind: str  # 'abs' | 'rel' | 'order' | 'prob'
    why: str


TOL: dict[str, Tol] = {
    # ------------------------------------------------------------------ parity
    "parity_model": Tol(
        1e-10, "abs",
        "Put-call parity C - P = S*exp(-q*tau) - K*exp(-r*tau) is an identity of the "
        "closed-form itself; the only error is float64 rounding across a handful of "
        "exp/mul ops, which stays well under 1e-10 in double precision.",
    ),
    "parity_market_resid_coin": Tol(
        0.05, "abs",
        "On real Deribit quotes, parity C-P vs (F-K)*df need only hold to within the "
        "bid-ask spread + discreteness. 0.05 coin (~5% of a typical near-ATM mark) is a "
        "reporting bound for flagging violations, NOT a pass/fail on model math; larger "
        "residuals are investigated (stale quote / wide spread) and written up, not hidden.",
    ),
    # ---------------------------------------------------- binomial -> BS limit
    "crr_bs_convergence_price": Tol(
        5e-3, "abs",
        "CRR European price at N=2000 steps vs Black-Scholes. CRR error is O(1/N) with "
        "the well-known oscillation; at N=2000 the envelope is a few 1e-3 for ATM unit-"
        "spot options. Tight enough to catch a real bug, loose enough for the oscillation.",
    ),
    "crr_convergence_order_min": Tol(
        0.85, "order",
        "Empirical convergence order of |CRR - BS| fit on a log-log Richardson ladder. "
        "CRR is first-order (order 1) with even-N averaging; we require the measured "
        "slope >= 0.85 so a broken lattice (order ~0) fails loudly. Measured value is "
        "reported, not just the pass.",
    ),
    # -------------------------------------------------------- Monte Carlo / CI
    "mc_ci_coverage_prob": Tol(
        0.90, "prob",
        "Across the price grid, the fraction of 95% MC confidence intervals that cover "
        "the closed-form price must be >= 0.90. With seeded 95% CIs the long-run rate is "
        "~0.95; 0.90 tolerates finite-grid noise while still failing a mis-scaled stderr.",
    ),
    "mc_ci_halfwidth_rel": Tol(
        0.02, "rel",
        "Sanity bound on MC precision at the configured path count: 95% CI half-width "
        "<= 2% of the ATM price. Ensures the variance-reduction machinery is actually "
        "engaged (a broken control variate would blow this up).",
    ),
    # --------------------------------------------------------- Greeks 3-way
    "greeks_fd_vs_closed_rel": Tol(
        1e-4, "rel",
        "Central finite differences vs closed-form Greeks. Central FD is O(h^2); at the "
        "per-Greek step sizes chosen (h tuned per variable) the truncation+rounding error "
        "sits near 1e-5..1e-4 relative. 1e-4 catches a wrong formula, passes a correct one.",
    ),
    "greeks_mc_vs_closed_rel": Tol(
        0.03, "rel",
        "MC pathwise (delta/vega) and likelihood-ratio estimators vs closed form. These "
        "are unbiased but noisy; 3% relative is ~2-3 standard errors at the configured "
        "path count. The estimator's own stderr is also reported so the gate is honest.",
    ),
    # ------------------------------------------------- American: LSMC vs lattice
    "lsmc_vs_crr_american_rel": Tol(
        0.01, "rel",
        "Longstaff-Schwartz American price vs a fine CRR lattice (the reference for "
        "American vanilla). LSMC has regression + MC error; 1% relative is the standard "
        "band in the literature at the configured paths/steps. Reported with the LSMC CI.",
    ),
    # ------------------------------------------------------- IV solver round-trip
    "iv_roundtrip_abs": Tol(
        1e-6, "abs",
        "Price a European at sigma0, invert with the solver, recover sigma. The solver "
        "converges to 1e-8 on price; the vol round-trip error is bounded by that over "
        "vega and stays under 1e-6 except in the documented deep-OTM/near-expiry regime.",
    ),
    # ----------------------------------------------------------- SVI calibration
    "svi_fit_rmse_report": Tol(
        0.01, "abs",
        "Reporting threshold (NOT pass/fail) for per-expiry SVI fit RMSE in total-variance "
        "space. Well-behaved liquid smiles fit to <0.01; larger RMSE is surfaced with the "
        "expiry and probable cause (sparse strikes, wide market) in DESIGN/RESEARCH_NOTE.",
    ),
    "svi_butterfly_g_min": Tol(
        -1e-8, "abs",
        "Gatheral's g(k) durrleman function must be >= 0 for a butterfly-arb-free slice. "
        "We require g >= -1e-8 (a float-noise band around 0); any strictly-negative g is "
        "reported as a constraint violation with its k location, never smoothed away.",
    ),
    # -------------------------------------------------------- exchange differential
    "exch_diff_report_only": Tol(
        0.0, "abs",
        "There is NO pass/fail tolerance on our-IV vs Deribit mark-IV: the mission forbids "
        "claiming agreement without printing the full distribution. This entry documents "
        "that the exchange differential is descriptive-only (median/IQR in vol points), "
        "with outliers investigated in writing.",
    ),
}


def summary() -> str:
    """Human-readable dump of the registry (used by tests/docs to print the table)."""
    lines = [f"{'name':<32} {'value':>12}  {'kind':<6} why"]
    for name, t in TOL.items():
        lines.append(f"{name:<32} {t.value:>12.2e}  {t.kind:<6} {t.why.splitlines()[0]}")
    return "\n".join(lines)
