# DESIGN.md — vol-lab modeling decisions & rationale

Every modeling choice in vol-lab (day count, rate assumption, illiquid-strike
filtering, SVI constraint set, tolerances) gets its rationale recorded here, per
CLAUDE.md. This is a Wave-0 skeleton: each section is a stub with a `TODO(Wx)`
marker naming the wave that fills it in. Numerical conventions referenced below
are frozen in `src/schema.py`, `src/interfaces.py`, and `config/tolerances.py`.

## Overview & scope
vol-lab prices vanilla crypto options and calibrates an implied-volatility
surface from real Deribit market data, cross-verifying four engines against each
other, against theory, and against the exchange's published mark IV. Strictly a
numerics/research project — no trading claims (see CLAUDE.md "Never do").

TODO(G4): final scope summary, engine/feature inventory, and headline results
once G1–G3 land.

## Market data & conventions
Source is the Deribit public REST API (no auth, ≥250ms request spacing, cached
under `data/snapshots/`; CI never calls live). BTC/ETH options are **inverse
contracts**: the premium is quoted in coin, so USD premium = coin premium ×
index price. Deribit `mark_iv` is a percent and is converted to a decimal
(65.0 → 0.65) to match the IV convention in the contracts.

TODO(W1): document the exact fixture schema, field provenance (`OptionQuote`),
index-vs-mark price sourcing, and the coin→USD conversion helper.

## Day count & rate assumption
Time to expiry uses ACT/365 (calendar days / 365) for the year fraction `tau`.
Working assumption is `r = q = 0` in USD forward terms, to be justified against
parity-inferred forwards rather than assumed away.

TODO(W2): justify ACT/365 vs ACT/365.25 on the snapshot horizons, and validate
the `r = q = 0` USD-forward assumption against the forwards backed out from
put-call parity (quantify the residual carry).

## Forwards from put-call parity
Forwards are **inferred**, not assumed. Put-call parity in USD terms is
`C(K) − P(K) = df · (F − K)`, so a linear regression of `C − P` on `K` has slope
`−df` and intercept `df · F`; hence `df = −slope`, `F = intercept / df`, and the
implied continuous rate is `r = −ln(df)/tau` (`src/surface/forwards.py`). Each
C/P pair is weighted by an inverse-relative-spread liquidity weight
`1/(1 + rs_C + rs_P)` so wide/stale strikes don't drag the fit; pairs are formed
only where both legs have a usable price (a real mid preferred, exchange mark as
fallback). The number of pairs used vs available is reported, never silently
dropped, and expiries with `< min_pairs` (default 3) return `None` rather than a
fabricated forward.

**External validation (fixture 2026-08-07, 1540 quotes).** The parity-inferred
forward matches Deribit's own published per-expiry `underlying_price` to within
**0.01–0.5%** across all 12 BTC and 12 ETH expiries (e.g. BTC 25Jun27:
F_parity 67,348 vs Deribit 67,311, +0.05%). This is a genuine external check on
the estimator, not a self-consistency loop. The implied `df ≈ 1` at short tenors
confirms the `r ≈ 0` USD-forward working assumption; the small positive forward
premium at long tenors is the crypto funding/carry, left in `df` rather than
assumed to zero.

## Pricing engines
- **BS** (`src/bs`) — closed-form Black-Scholes-Merton with carry; the reference for European prices/Greeks.
- **CRR** (`src/lattice`) — Cox-Ross-Rubinstein binomial; European convergence to BS (measured order) and American exercise.
- **MC** (`src/mc`) — seeded Monte Carlo with confidence intervals and variance reduction; European cross-check.
- **LSMC** (`src/lsmc`) — Longstaff-Schwartz least-squares Monte Carlo for American options, benchmarked to a fine CRR lattice.

TODO(G1/G3): per-engine algorithm notes, discretization/step choices, and the
measured cross-engine differentials (convergence order, MC CI coverage, LSMC vs CRR).

## IV solver
Implied vol is solved from price via a bracketed root find with a Newton step and
a vega floor to stay stable near zero vega. Returns `None` (honest unknown)
rather than a plausible-but-wrong number when it cannot bracket/converge.

TODO(G1): document the bracket construction, Newton/bisection fallback logic, the
vega-floor value, and the enumerated failure modes (deep OTM, near-expiry, price
outside no-arbitrage bounds).

## SVI parameterization & no-arbitrage constraints
Per-expiry smiles are fit with **raw SVI** (Gatheral 2004) in total-variance
space as a function of log-moneyness `k = ln(K/F)`:
`w(k) = a + b·(rho·(k−m) + sqrt((k−m)² + sigma²))` (`src/surface/svi.py`).
Parameters: `a` level, `b ≥ 0` wing angle, `−1 < rho < 1` skew, `m` horizontal
shift, `sigma > 0` curvature.

**Constraint set (enforced during the fit, SLSQP):**
- `b ≥ 0`, `−0.999 ≤ rho ≤ 0.999`, `sigma ≥ 1e−4` (domain bounds);
- `a + b·sigma·sqrt(1−rho²) ≥ 0` — the global minimum of `w(k)` stays
  non-negative (no negative variance anywhere);
- `b·(1+|rho|)·tau ≤ 4` — the Lee moment bound on asymptotic wing slopes
  (no vertical-spread arbitrage in the tails).

Calibration is a **deterministic multi-start** (fixed starts over `rho`/`sigma`;
no RNG) weighted least squares, with real mids weighted `1.0` vs mark-fallbacks
`0.3`. Butterfly no-arbitrage (`g(k) ≥ 0`, `TOL["svi_butterfly_g_min"]`) and
calendar monotonicity are validated **after** the fit by `src/noarb` and reported
per surface — never smoothed into the objective silently. RMSE is reported in
both total-variance and vol-point space; `TOL["svi_fit_rmse_report"]` (0.01) is a
*reporting* threshold, not pass/fail — slices above it are surfaced with cause.

**Fit quality (fixture 2026-08-07).** Per-expiry `rmse_w` is `~5e−5` to `~1e−3`
for the liquid tenors on both BTC and ETH.

**Outlier investigated — 25Sep26 (`rmse_w ~6–7e−3`, both coins).** Not smoothed
away; investigated: this slice carries a long deep-OTM-call wing (log-moneyness up
to `k ≈ +1.6` BTC / `+2.1` ETH) where no two-sided market exists, so those points
fall back to the exchange **mark** IV. The mark on those far strikes is *pinned* —
BTC sits flat at 75.4% across `k ∈ [0.90, 1.31]`, ETH flat at 93.4% across
`k ∈ [1.05, 1.36]` — a stale-mark shelf the exchange isn't refreshing. A smooth
SVI arch cannot match a flat plateau glued to a rising wing, which inflates the
RMSE. The mark-fallback down-weighting (0.3 vs 1.0) already limits the shelf's pull
on the fit; the liquid core (real mids, `|k| ≲ 0.7`) fits cleanly. This is a
*data-quality* artifact of illiquid far-wing marks, not a model failure — the
honest read is that far-wing mark IV is unreliable, which is itself a finding the
exchange-differential (`src/exchdiff`) quantifies.

## Liquidity filtering
Illiquid strikes are filtered by minimum open interest and maximum relative
bid-ask spread before calibration. Filtering is **never silent**: the count kept
vs dropped and the thresholds are reported alongside every surface.

TODO(G2): fix the min-OI / max-spread thresholds with rationale, and define the
per-expiry filter-stats block emitted with each calibration.

## Tolerance registry
All differential-test tolerances live in `config/tolerances.py` (`TOL` dict),
each with a written justification. Tests reference them by name (e.g.
`TOL["parity_model"].value`); tolerances are never widened to turn red green —
a failure is investigated and written up here.

TODO(G1+): as each gate lands, cross-reference the specific `TOL` entries it
exercises and record any investigation that a near-threshold result triggered.

## Determinism & reproducibility
All randomness is seeded (default seed 12345, seed accepted as a parameter);
every published statistic is reproducible by one documented command. Dev machine
is Apple Silicon, CI is ubuntu-latest, both on pinned deps (see `docs/ENV.md`).

TODO(G3): record the exact reproduce commands per figure/statistic in the
research note and confirm cross-platform (arm64 dev vs x86-64 CI) determinism.
