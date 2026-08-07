# DESIGN.md — vol-lab modeling decisions & rationale

Every modeling choice in vol-lab (day count, rate assumption, illiquid-strike
filtering, SVI constraint set, tolerances) gets its rationale recorded here.
Numerical conventions referenced below are frozen in `src/schema.py`,
`src/interfaces.py`, and `config/tolerances.py`; measured results cite the
2026-08-07 snapshot fixtures and are reproducible via `scripts/report_surface.py`.

## Overview & scope
vol-lab prices vanilla crypto options and calibrates an implied-volatility
surface from real Deribit market data, cross-verifying four engines against each
other, against theory, and against the exchange's published mark IV. Strictly a
numerics/research project — no trading claims, no forecasts (the research note is
descriptive only, with confidence intervals).

**Delivered:** four pricing engines (Black-Scholes, CRR binomial, Monte Carlo,
Longstaff-Schwartz), a Deribit snapshot pipeline (1,540 BTC+ETH instruments),
parity-inferred forwards, per-expiry SVI calibration under no-arbitrage
constraints, a no-arb scanner, an exchange differential vs mark IV, a descriptive
research note, and 8 script-generated figures. Headline verification: CRR→BS
convergence order 0.9993, Greeks three-way to `<0.5%`, parity to ~2e-14, exchange
median |Δσ| 0.18/0.39 vol pts, 96% coverage, green CI. See `README.md` for the
stats table.

## Market data & conventions
Source is the Deribit public REST API (no auth, ≥250ms request spacing, cached
under `data/snapshots/`; CI never calls live). BTC/ETH options are **inverse
contracts**: the premium is quoted in coin, so USD premium = coin premium ×
index price. Deribit `mark_iv` is a percent and is converted to a decimal
(65.0 → 0.65) to match the IV convention in the contracts.

Fixtures are raw exchange JSON captured by `scripts/collect_snapshot.py` (one
`get_book_summary_by_currency` call returns the full board per coin) and parsed into
the frozen `OptionQuote`/`Snapshot` contracts by `src/deribit/store.py`. Each
`OptionQuote` carries full provenance — instrument name, snapshot timestamp, index
price, per-instrument underlying (Deribit's forward), and both the coin and USD marks —
so any statistic traces back to the exact quote. `mark_price_usd = mark_price_coin ×
index_price` is the inverse-contract conversion; both are retained so it is auditable.

## Day count & rate assumption
Time to expiry uses ACT/365 (calendar days / 365) for the year fraction `tau`.
Working assumption is `r = q = 0` in USD forward terms, justified against
parity-inferred forwards rather than assumed away. ACT/365 (vs ACT/365.25) is a
`<0.07%` difference in `tau` even at the ~11-month horizon — negligible next to
bid-ask noise — so the simpler convention is used. The `r = q = 0` USD-forward
assumption is *not* imposed: the parity regression backs out `df` per expiry, and
the observed `df ≈ 1` at short tenors (with a small funding/carry premium at long
tenors) confirms it holds to within the residual reported in Forwards below.

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
**~0.7%** across all 12 BTC and 12 ETH expiries — most under 0.3% (e.g. BTC 25Jun27:
F_parity 67,348 vs Deribit 67,311, +0.05%); the sole outlier is one thin far-dated ETH
line (25Jun27) at +0.69%. This is a genuine external check on
the estimator, not a self-consistency loop. The implied `df ≈ 1` at short tenors
confirms the `r ≈ 0` USD-forward working assumption; the small positive forward
premium at long tenors is the crypto funding/carry, left in `df` rather than
assumed to zero.

## Pricing engines
- **BS** (`src/bs`) — closed-form Black-Scholes-Merton with carry; the reference for European prices/Greeks.
- **CRR** (`src/lattice`) — Cox-Ross-Rubinstein binomial; European convergence to BS (measured order) and American exercise.
- **MC** (`src/mc`) — seeded Monte Carlo with confidence intervals and variance reduction; European cross-check.
- **LSMC** (`src/lsmc`) — Longstaff-Schwartz least-squares Monte Carlo for American options, benchmarked to a fine CRR lattice.

**Measured cross-engine differentials (G1/G2):** CRR→BS convergence order **0.9993**
(log-log fit over N = 50…1600, even-N to avoid the odd/even sawtooth; error halves per
N-doubling); MC 95% CIs cover the closed form on the full grid; Greeks closed-form vs
central FD `<1e-4` rel, vs MC pathwise/LR `<0.5%` rel; LSMC American vs CRR(2000)
`max 0.43%` rel (converges from below with 50 exercise dates / 100k paths). CRR uses
`u = e^{σ√dt}`, `d = 1/u`, `p = (e^{(r−q)dt}−d)/(u−d)`, default 2000 steps; MC uses an
exact terminal lognormal step (no Euler bias) with antithetic + control variate.

## IV solver
Implied vol is solved from price via a bracketed root find with a Newton step and
a vega floor to stay stable near zero vega, with a Brent bisection fallback when
Newton escapes the bracket or vega collapses (`src/bs`). Converges price to ~1e-8.

**Enumerated failure modes — returns `None` (honest unknown), never a fabricated vol:**
`tau ≤ 0` (expired); price below discounted intrinsic (sub-intrinsic arbitrage); price
at/above the no-arb ceiling (`call ≥ S·e^{−qτ}`, `put ≥ K·df`, i.e. `σ→∞`); and the
deep-OTM/near-expiry regime where vega underflows to ~0 (e.g. `K=300, tau=0.002`) so no
vol is recoverable. In the recoverable-but-hard deep-OTM/short-tau case the solver hits
the vega floor, falls back to bisection, and still converges *price* to `<1e-8` even where
*vol* resolution loosens — tested on price, documented on vol.

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
- `b·(1+|rho|) ≤ 2` — the Lee moment bound on the asymptotic wing slope of *total*
  variance (each wing slope ≤ 2; τ-independent because raw SVI's `w` is already
  σ²·τ), ruling out vertical-spread arbitrage in the tails.

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

## No-arbitrage scan (`src/noarb`)
Three static-arbitrage checks, run on the calibrated surface (and price-convexity on a
reconstructed call curve), all **quantified, never smoothed**:
- **Butterfly (slice):** Gatheral's Durrleman `g(k)` from *analytic* raw-SVI derivatives
  `w'`, `w''`; flags `g(k) < TOL["svi_butterfly_g_min"]` (−1e−8) over a dense ±1.5 k-grid,
  reporting the min `g` and its `k`.
- **Butterfly (price):** convexity of the undiscounted call curve in strike across
  adjacent triplets (right-slope − left-slope ≥ 0).
- **Calendar:** total variance non-decreasing in `tau` at fixed `k`, sampled on the
  *overlap* of adjacent calibrated k-ranges (no extrapolation).

**Result (fixture 2026-08-07).** Calendar arbitrage: **zero** on the snapshot, both
coins. Price-convexity violations are tiny (≤ 0.12 USD/strike), consistent with
bid-ask/discreteness noise. The one SVI `g<0` flag per surface sits at `k ≈ 0.5–0.8`
— **outside the liquid strike range** (`|k| ≲ 0.2`) on a short-tau slice, i.e. a
wing-*extrapolation* artifact of near-expiry SVI, not traded-region arbitrage. The scan
grid deliberately spans past the wings so this extrapolation risk is surfaced, not hidden.

## Exchange differential (`src/exchdiff`)
The external verifier the mission is built around: our independently-written IV solver's
vols vs Deribit's **published mark IV**. For each option with both a valid mark IV and a
usable mid, `Δσ = our_iv − mark_iv` (vol points). Distribution summarized by median, IQR,
mean, std, and median-absolute — overall and bucketed by moneyness (ATM/near/wing) and
expiry (short/medium/long). Descriptive only (`TOL["exch_diff_report_only"]`): we never
claim "agreement" without printing the distribution. Outliers are ranked with a diagnosed
cause (`mark-fallback` > `wide-spread` > `deep-OTM wing` > `mark-IV construction`).

**Result (fixture 2026-08-07).** Median `|Δσ|` = **0.18 (BTC) / 0.39 (ETH) vol points**
(range 0.11–0.45 across the two snapshots; BTC tighter than ETH). Residuals grow toward
ATM/near-dated (BTC ATM ≈ 0.22–0.89 vp) exactly where mark-IV timing/smoothing bites hardest, and are
smallest in the liquid wings — the honest, expected pattern. Largest outliers are
next-day ATM strikes (mark timing) and wide-spread far-wing calls (vega collapse). This
is strong evidence our solver is correct *and* a descriptive read on how Deribit builds
its mark IV — not a tuned match.

## Liquidity filtering
Two liquidity mechanisms, both non-silent:
1. **Hard filters** (`src/deribit/filters.py`): optional minimum open interest and
   maximum relative bid-ask spread, defaulting to no-op so callers opt in. Every
   dropped quote is recorded in `FilterStats` with a first-failing reason
   (`low_open_interest` > `no_rel_spread` > `wide_spread`), and `FilterStats.check()`
   asserts `n_in == n_out + n_dropped` so nothing vanishes unaccounted.
2. **Soft weighting** (used in the surface): rather than hard-cutting the wings, the
   forward regression and SVI fit *weight* each point — inverse relative spread for
   forwards, and `1.0` for real two-sided mids vs `0.3` for exchange-mark fallbacks in
   SVI. This keeps illiquid far strikes visible (so the no-arb scan and exchange
   differential can report on them) without letting them dominate the fit. Rationale:
   silently dropping the wings would hide exactly the stale-mark artifacts the project
   is meant to surface (see the 25Sep26 outlier above).

## Tolerance registry
All differential-test tolerances live in `config/tolerances.py` (`TOL` dict),
each with a written justification. Tests reference them by name (e.g.
`TOL["parity_model"].value`); tolerances are never widened to turn red green —
a failure is investigated and written up here.

Gate → tolerance map: **G1** exercises `parity_model` (~2e-14 ≪ 1e-10),
`crr_convergence_order_min` (measured 0.9993 ≥ 0.85), `crr_bs_convergence_price`,
`mc_ci_coverage_prob` (6/6), `mc_ci_halfwidth_rel`, `greeks_fd_vs_closed_rel`,
`greeks_mc_vs_closed_rel` (<0.5%), `iv_roundtrip_abs`. **G2** exercises
`lsmc_vs_crr_american_rel` (0.43% < 1%), `svi_butterfly_g_min`, `svi_fit_rmse_report`
(reporting-only; the 25Sep26 slice exceeded it and was investigated above), and
`exch_diff_report_only` (descriptive, no pass/fail). No tolerance was widened at any
gate; the only near-threshold event (the 25Sep26 RMSE) was investigated in writing
(stale far-wing mark), not resolved by moving the bound.

## Determinism & reproducibility
All randomness is seeded (default seed 12345, seed accepted as a parameter);
every published statistic is reproducible by one documented command. Dev machine
is Apple Silicon, CI is ubuntu-latest, both on pinned deps (see `docs/ENV.md`).

**Reproduce commands:**
- Every surface/verification statistic: `python scripts/report_surface.py --all-snapshots`
  (verified byte-identical across repeated runs).
- Every figure: `python scripts/make_figures.py` → `docs/figures/` (content-deterministic;
  most are md5-identical across regeneration, a few differ only in matplotlib/Agg
  rasterization noise, not in the underlying data).
- Full verification suite: `coverage run -m pytest && coverage report`.
- A fresh snapshot: `python scripts/collect_snapshot.py` (polite public API).

Cross-platform: the numeric gates pass identically on the arm64 dev machine and
ubuntu-latest CI (green on every gate push); floating-point differentials are set well
above cross-platform ULP noise by the tolerance registry, so the tests are stable across
both.
