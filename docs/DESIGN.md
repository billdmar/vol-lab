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
Forwards are **inferred**, not assumed: for each expiry, `F` (and the implied
discount) are backed out from put-call parity on the most liquid strikes rather
than taken from an index or a fixed-rate curve.

TODO(W2): fix the estimator (regression of `C − P` on `K`, liquid-strike
weighting, per-expiry robustness) and report inferred `F`/`df` with diagnostics.

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
Per-expiry smiles are fit with the raw SVI parameterization in total-variance
space, subject to no-arbitrage constraints: butterfly (Gatheral's `g(k) ≥ 0`) and
calendar (non-decreasing total variance) monotonicity.

TODO(W2/G2): state the exact SVI form and parameter bounds, the constraint set
and how it's enforced during calibration, and the RMSE reporting convention
(`TOL["svi_fit_rmse_report"]`).

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
