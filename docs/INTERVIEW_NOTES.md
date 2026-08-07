# vol-lab — Options Market-Maker Interview Notes

Fifteen screen-standard questions (Akuna / SIG / Optiver / IMC / CTC), each answered
**from this codebase** with the file that implements it and the number it produced.
Every figure below was regenerated from the committed fixtures
(`data/snapshots/snapshot_20260807_*.json`, BTC + ETH, 1540 quotes each) via
`python scripts/report_surface.py --all-snapshots` and the G1/LSMC test suites
(`pytest -s`). Nothing here is a forecast or a trading claim — the project prices
and reconciles, it does not predict.

---

## 1. State put-call parity. How did vol-lab use it?

Parity in USD/forward terms: `C(K) − P(K) = df · (F − K)`, where `df = e^{−rτ}` is the
discount factor and `F` the forward. This is a model-free identity — no vol, no
distributional assumption — so I used it *backwards* to infer the forward instead of
assuming one. In `src/surface/forwards.py`, `infer_forward` regresses the USD call-minus-put
difference on strike across every liquid C/P pair (weighted by inverse relative spread):
the slope is `−df` and the intercept is `df·F`, hence `df = −slope`, `F = intercept/df`.

Validation: the inferred `F` is cross-checked against Deribit's own published
`underlying_price` per expiry. Across all 12 expiries × 2 currencies × 2 snapshots the
match `dF%` is within roughly 0.5% (most are under 0.3%; the only outliers are
thin far-dated ETH lines, e.g. one 25-Jun-27 expiry at +0.69%). On *model* prices parity
holds to machine precision (see Q4/Q9). The point I make in interview: the forward is
*measured*, and I can defend the residual against the exchange's own number.

## 2. Walk each Greek (sign + meaning). How did three implementations reconcile?

- **Delta** `∂P/∂S`: hedge ratio; +for calls (0→e^{−qτ}), −for puts.
- **Gamma** `∂²P/∂S²`: convexity of delta; +for long options, peaks ATM.
- **Vega** `∂P/∂σ`: +for long options, peaks ATM, grows with `√τ`.
- **Theta** `∂P/∂t`: time decay; −for long options (the diffusion term `−Sn(d₁)σ/(2√τ)` in
  `src/bs/black_scholes.py:greeks`).
- **Rho** `∂P/∂r`: +for calls, −for puts.

Three independent engines agree in the G1 gate (`tests/test_integration_g1.py`):
closed-form Black-Scholes (`src/bs`) vs **central finite differences** vs **Monte Carlo**
(`src/mc`). Measured at ATM, τ=0.75, σ=0.65, r=q=0: FD-vs-closed relative error is ~1e-9
(well under the 1e-4 gate); MC **pathwise** delta = 0.13% and vega = 0.24%; MC
**likelihood-ratio** gamma = 0.45%. Pathwise differentiates the payoff through
`dS_T/dS = S_T/S` (low variance, but only legal because the vanilla payoff is a.s.
differentiable); gamma uses the LR score weight `(Z²−1−σ√τ·Z)/(S²σ²τ)` because the
second derivative of a kinked payoff can't be taken pathwise. Each MC Greek reports its
own standard error so the reconciliation is honest about sampling noise.

## 3. Why does the smile/skew exist? What does vol-lab observe?

Black-Scholes assumes lognormal returns; real return distributions have **fat tails and
asymmetry**, so out-of-the-money options are worth more than a flat vol implies — the
market prices in crash/jump risk, and demand for protection tilts the wings. That shows
up as a smile (curvature = butterfly `BF25`) and a skew (tilt = risk-reversal `RR25`).

Observed (from `report_surface.py`):
- **BTC** carries a persistent **downside skew** that *steepens with tenor*: `RR25`
  runs from about −0.7 to −1.8 vol pts at the front to −4.0 to −4.6 vol pts at 2–5
  months — downside puts bid richer than upside calls, more so further out.
- **ETH** is the more interesting story: the skew **flips sign inside the first week**.
  The 0–2-day expiries show a *positive* `RR25` (+1.9, +2.6 — call/upside skew), it
  crosses through zero at the 10-Aug expiry (≈−0.06) into downside by 11-Aug (−1.5), and
  every tenor beyond is downside (to about −2.9). I read the front-end call skew
  descriptively as short-dated upside demand, not as a signal.

## 4. What is implied vol? How does your solver work, and how can it fail?

IV is the single σ that makes the Black-Scholes price equal the observed price — it's the
market's quote *in vol units*, not a prediction. `BlackScholes.implied_vol`
(`src/bs/black_scholes.py`) inverts the price monotone-in-σ: it first checks the no-arb
envelope, then brackets σ so the price is straddled, then runs **safeguarded Newton
(rtsafe)** — Newton steps when vega is healthy, bisection when a step escapes the bracket
or vega drops below a floor — with **Brent** as a final bracketed fallback. It converges the
*price* to ~1e-8.

It **returns `None` (honest-unknown, never a fabricated vol)** when no arbitrage-free IV
exists: `τ ≤ 0`; price below discounted intrinsic (sub-intrinsic arbitrage); price at/above
the no-arb ceiling (`C ≥ Se^{−qτ}`, `P ≥ Ke^{−rτ}`, σ→∞); or price pinned at intrinsic
within float resolution — the **deep-OTM / near-expiry vega collapse**, where the option
carries no recoverable vol and returning a meaningless ~0 would be a lie. That last case is
exactly the wing behavior I see in the exchange differential (Q11).

## 5. American vs European: when is early exercise optimal? LSMC intuition?

Early exercise is optimal when the intrinsic beats the continuation value — for a **put**
when it's deep ITM and rates are high (exercising frees cash that earns `r`); for a
call only with a dividend/carry large enough to offset the forfeited time value. The CRR
lattice (`src/lattice/crr.py`) takes `max(continuation, intrinsic)` at every node; the G1
gate measures a strictly positive **early-exercise premium of ≈1.69** on an ITM
high-rate American put (American 14.50 vs European 12.80).

**LSMC** (`src/lsmc/lsmc.py`, Longstaff-Schwartz) is the Monte-Carlo route: simulate full
GBM paths, then walk *backward* regressing the discounted future cashflow on a
degree-3 polynomial of the spot **on in-the-money paths only** — the fitted value is the
estimated continuation, and you exercise where intrinsic exceeds it. ITM-only regression is
the standard variance choice (OTM paths carry no exercise decision). Cross-checked against a
2000-step CRR lattice, the LSMC American-put price lands within **max 0.43%** across four
ITM/ATM cases (tolerance 1.00%), each with a reported 95% CI.

## 6. Variance reduction in MC — techniques and measured speedup?

Two techniques in `src/mc/engine.py`, both toggleable so `price` keeps the frozen signature:
- **Antithetic** variates: draw `Z`, reuse `−Z`; the stderr is computed on the *pair means*
  so it stays statistically honest.
- **Control variate**: the discounted terminal spot `df·S_T` has known exact mean
  `S·e^{−qτ}` and is strongly correlated with the payoff, so subtracting the
  optimal-β-scaled control deviation cuts variance while staying unbiased.

`variance_reduction_report` measures the equal-precision path-count multiplier (ratio of
estimator variances at the same seed and paths). At the test's base contract
(S=K=100, τ=0.75, r=3%, σ=65%, q=1%, 100k paths): **control-alone 8.5×**, **full combo
6.4×**, antithetic ~1.3×. Honest note I always give: **control-alone beat the combo here**
— stacking antithetic on top slightly *raised* variance because the antithetic pairing
weakens the payoff/control correlation the control variate depends on. More technique isn't
automatically more reduction, and I can show the number that proves it.

## 7. What is SVI and why parameterize the smile? Raw-SVI + no-arb constraints.

SVI (Gatheral) fits the whole smile with 5 parameters instead of storing per-strike vols,
giving a smooth arb-checkable curve you can interpolate/extrapolate. **Raw SVI** on total
implied variance (`src/surface/svi.py`):

`w(k) = a + b·(ρ·(k−m) + √((k−m)² + σ²))`, `k = ln(K/F)`

`a` level, `b ≥ 0` wing angle, `ρ∈(−1,1)` skew, `m` horizontal shift, `σ` curvature at the
minimum. No-arb discipline in the calibration bounds/constraints:
- **`w(k) ≥ 0` globally** via `a + b·σ·√(1−ρ²) ≥ 0` (the analytic global minimum).
- **Lee wing bound**: `b·(1+|ρ|)·τ ≤ 4` caps the asymptotic wing slope at 2 per side.
- **Butterfly** is checked *after* the fit with Durrleman's `g(k) ≥ 0` (Q10), never
  smoothed away.
- **Calendar** monotonicity (`w` non-decreasing in τ) checked across expiries (Q10).

Fit is deterministic (fixed multi-start over ρ/σ, no RNG); RMSE is reported in both
total-variance and vol-point space (e.g. `rmse_w` ~5e-5 to 8e-3 across slices).

## 8. Inverse contracts: how are Deribit crypto options quoted, and how did you convert?

Deribit BTC/ETH options are **inverse**: the premium is quoted **in the coin**, not USD. To
get a USD premium you multiply by the index: `USD = coin_premium × index_price`
(`src/deribit/store.py`; used in `forwards.py:_usd_price`). Both representations are
retained through the pipeline so nothing is lost, and the parity regression / pricing all
run in USD forward terms. This is the crypto-substrate detail screeners like to probe, and
the convention is documented in `docs/DESIGN.md`.

## 9. How do you infer the forward without assuming a rate?

Same parity regression as Q1 — and critically it delivers the rate too, not just `F`. From
`df = −slope` I back out a continuous rate `r = −ln(df)/τ` (`forwards.py`, returns 0 when
`df ≈ 1`, the crypto default). So I never *assume* a risk-free rate: I measure the discount
factor the market is actually pricing and report the implied `r`. The regression uses only
strikes where **both** legs have a real two-sided mid (falling back to the mark only when no
mid exists), and drops expiries with fewer than 3 usable pairs rather than fabricate a
forward from one strike.

## 10. No-arbitrage: butterfly and calendar — how do you test them, and what did you find?

`src/noarb/scan.py`, three model-free/parametric checks, all **quantified, never smoothed**:
- **Butterfly (SVI)**: Durrleman `g(k) = (1 − kw'/2w)² − (w'²/4)(1/w + 1/4) + w''/2` on a
  dense k-grid with analytic derivatives; `g(k) ≥ 0` is arb-free.
- **Butterfly (raw prices)**: undiscounted call prices must be convex in strike (long
  butterfly ≥ 0).
- **Calendar**: total variance non-decreasing in τ at fixed k, sampled on the *overlap* of
  adjacent expiries' calibration ranges.

Findings across both snapshots: **0 calendar violations** (both currencies) — the term
structure of variance is monotone. Each surface shows exactly **one SVI slice with a
`g_min < 0`**, and its location (`k ≈ 0.45–0.80`, a short-dated expiry's wing) marks it as a
**wing-extrapolation artifact** — the fit is convex through the traded strikes and only dips
negative out where the calibration grid runs past the data. A handful of raw price-convexity
"violations" appear (27–47, worst ≈0.01–0.12 USD), consistent with bid/ask granularity, not
structural arbitrage. I flag every one; I never widen a tolerance to hide it.

## 11. What did the exchange differential teach you about your solver *and* mark-IV construction?

`src/exchdiff/differential.py` compares my solved IV against Deribit's published mark IV on
every matched strike. Median `|Δσ|` is **0.18 vol pts (BTC) / 0.39 (ETH)** (range 0.11–0.45
across the two snapshots) — close, and I always show the *distribution* (median, IQR,
by-moneyness, outliers), never just claim "agreement." Two lessons the residual structure
taught me:
- **Residuals peak at near-dated ATM** (ATM median `|Δσ|` ≈ 0.8–0.9 vol pts vs ~0.1 in the
  wings). ATM is where vega is largest, so a tiny mark-price rounding or a smoothing/timing
  difference in *how the exchange builds its mark* maps to a visible vol gap — this is about
  **mark construction**, not a solver error.
- **The wings are noisiest in an absolute-price sense but smallest in `|Δσ|`** because vega
  collapses there — exactly the regime where my solver returns `None` rather than a bogus
  vol (Q4). Seeing my honest-unknown boundary line up with the empirically hardest strikes
  is the validation I care about.

## 12. Convergence: what order is the binomial tree, and how did you measure it?

CRR error is **O(1/N)** (first order). `convergence_order` (`src/lattice/crr.py`) prices the
European option at a ladder of step counts, then fits a line to `log|CRR(N) − BS|` vs
`log(N)`; the negated slope is the empirical order. Measured order = **0.9993** (gate 0.85),
with errors halving cleanly per doubling (3.97e-2 → 1.99e-2 → 9.95e-3 → …). The one subtlety
I'd raise on a whiteboard: CRR error oscillates with the **parity of N** (the odd/even
sawtooth, from whether a tree node lands on the strike), so the ladder uses **even N only**
to sample one clean branch of the envelope and get a stable ~1.0 slope.

## 13. Term structure of vol: what shape, and what do contango/backwardation mean?

**Contango** = upward-sloping (longer-dated vol > short-dated); **backwardation** = the
reverse, typically after a shock when near-term realized vol spikes. Both snapshots show
clean **contango** in ATM vol: **BTC ≈ 25% → 42%** and **ETH ≈ 34% → 56%** from the front
weeklies out to the 25-Jun-27 expiry (`Surface.term_structure`, printed by
`report_surface.py`). I describe this descriptively — an upward vol term structure with the
front pinned low — and explicitly do **not** forecast which way it moves.

## 14. Vega/gamma/theta and the P&L of a delta-hedged option.

A delta-hedged long option is flat in spot to first order, so its P&L is driven by the
**gamma-theta tradeoff**. Over `dt` the position earns from realized moves via gamma —
`½·Γ·(dS)²` — and pays time decay `Θ·dt`. Because for a long option **gamma > 0** and
**theta < 0** (the `−Sn(d₁)σ/(2√τ)` decay term in `BS.greeks`), the delta-hedger is *long
gamma, short theta*: it profits when realized variance exceeds what theta charges and bleeds
otherwise. The rough breakeven is realized move ≈ implied — i.e. the daily move needed for
gamma gains to offset theta is set by the option's own IV. Vega measures exposure to the
*level* of implied vol repricing the still-open position. This is a descriptive read of the
Greeks I implemented; I make no P&L or strategy claim.

## 15. Honest limitations — what can't this project claim?

- **No trading edge, ever.** No PnL, no strategy, no alpha, no forecast. The research note is
  descriptive with stated uncertainty.
- **1-day data window so far.** Only two intraday snapshots (~40 min apart, 2026-08-07)
  are committed, so any **IV-vs-realized** or time-series claim is out of scope — I'd need
  many days and I refuse to fabricate them.
- **Crypto substrate.** BTC/ETH inverse options on one venue (Deribit); the skew/term-structure
  observations are about *this* market, not equities or a universal claim.
- **Static, descriptive scope.** No dynamic hedging simulation, no exotic payoffs, no
  live/CI API calls.

The framing I lead with: **"pricing has right answers, so I built where I can be proven
wrong."** Cross-engine differentials, measured convergence order, parity to machine
precision, and an honest-unknown IV solver are all *falsifiable* — that's the point of the
project, and it's why every number above regenerates from one command.

---

## Closing: what the screens add that this project can't drill

The project demonstrates derivatives *modeling* judgment; the screens also test raw speed and
probability instinct under a clock. Round out with:
- **Mental math** — sub-8-second arithmetic (Optiver/IMC 80-in-8 style): fast
  multiplication, %-of-large-numbers, fractions↔decimals, sequences.
- **Expected value & fair-price games** — EV of dice/coin/card bets, "would you take this
  bet," market-making a number on an unknown quantity and updating on a counterparty's trade.
- **Combinatorics & basic probability** — conditional probability, Bayes, binomial, birthday-
  paradox-style estimates, gambler's-ruin intuition.
- **Sequential/strategy puzzles** — pirate/hat/prisoner problems and simple game-theory
  (backward induction — which, notably, is the same logic as the American-exercise and LSMC
  recursions in Q5).
