# vol-lab — Interview Cheat Sheet

One page to glance at before a screen. Full answers: [`INTERVIEW_NOTES.md`](INTERVIEW_NOTES.md).
Numbers reproduce via `python scripts/report_surface.py --all-snapshots`.

## 60-second pitch
> An options-pricing and implied-vol surface engine on **real Deribit crypto-options data**.
> Four engines — Black-Scholes, CRR binomial, Monte Carlo, Longstaff-Schwartz — cross-verified
> against each other, against theory, and against the exchange's published mark IV, with SVI
> calibration under no-arbitrage constraints. *Pricing has right answers, so I built the
> project where I can be proven wrong* — descriptive only, no trading claims.

## Headline numbers (recall cold)
| Metric | Value |
|---|---|
| Instruments | **1,540** (BTC 830 + ETH 710), 12 expiries each |
| Binomial → BS convergence order | **0.9993** (measured, first-order) |
| Put-call parity (model) | **~2e-14** (machine precision; gate 1e-10) |
| Greeks 3-way | closed vs FD **<1e-4**; vs MC **<0.5%** |
| MC variance reduction | **8.5×** control / 6.4× combo / 1.3× antithetic¹ |
| LSMC vs fine CRR (American) | **0.43%** (gate 1%) |
| Exchange differential (our IV vs mark IV) | median \|Δσ\| **0.18** BTC / **0.39** ETH vol pts² |
| Forward match vs Deribit's own | within **~0.7%** (max 0.69%, one far-dated ETH line) |
| No-arbitrage | **0 calendar** violations; butterfly clean in liquid region |
| Term structure (ATM, contango) | BTC ~**26→42%** · ETH ~**34→56%** |
| Tests / coverage | **201** tests, **96%** engine coverage, mypy-clean, CI green |
| Tolerance registry | **13** named tolerances, each justified, never widened |

¹ at S=K=100, τ=0.75, r=3%, σ=65%, q=1%, 100k paths (parameter-dependent).
² mean of the 2 snapshots (per-snap BTC 0.11/0.25, ETH 0.32/0.45).

## Formula flash-cards (all verified against the code)
- **Put-call parity:** `C − P = df·(F − K)`, with `df = e^{−rτ}`, `F = spot·e^{(r−q)τ}`.
  Used to *infer* the forward: regress `C−P` on `K` → `df = −slope`, `F = intercept/df`.
- **Black-Scholes:** `d1 = [ln(F/K) + ½σ²τ] / (σ√τ)`, `d2 = d1 − σ√τ`;
  `C = df·(F·N(d1) − K·N(d2))`, `P = df·(K·N(−d2) − F·N(−d1))`.
- **Greeks (signs/units):** δ = ∂P/∂S; γ = ∂²P/∂S²; vega = ∂P/∂σ (per 1.00 vol);
  θ = ∂P/∂t (per **year**, <0 for long options); ρ = ∂P/∂r.
- **CRR:** `u = e^{σ√dt}`, `d = 1/u`, `p = (e^{(r−q)dt} − d)/(u − d)`; American =
  `max(continuation, intrinsic)` per node; **raise if p ∉ [0,1]** (arbitrageable tree).
- **Raw SVI (total variance):** `w(k) = a + b·(ρ(k−m) + √((k−m)² + σ²))`, `k = ln(K/F)`,
  `IV = √(w/τ)`.
- **No-arb constraints:** global min `a + b·σ·√(1−ρ²) ≥ 0` (w ≥ 0 everywhere);
  **Lee wing bound `b(1+|ρ|) ≤ 2`** (τ-independent, on total variance);
  butterfly = Gatheral's Durrleman `g(k) ≥ 0`; calendar = `w` non-decreasing in τ.
- **MC control variate:** control = `df·S_T`, mean `spot·e^{−qτ}` (analytic ⇒ unbiased),
  β = cov/var; antithetic pairs `[Z, −Z]`, stderr on **pair-means**.
- **LR gamma weight:** `(Z² − 1 − σ√τ·Z) / (S²σ²τ)` (Glasserman).

## 18-question skeletons (trigger → key point → file; full answers in INTERVIEW_NOTES)
1. **Parity** → identity above; used to infer forwards → `src/surface/forwards.py`.
2. **Greeks 3-way** → closed vs FD (<1e-4) vs MC (<0.5%) → `tests/test_integration_g1.py`.
3. **Why the smile** → non-lognormal tails/skew; BTC downside, ETH flips → `RESEARCH_NOTE §3`.
4. **IV solver + failure** → bracket+Newton+vega-floor+Brent; `None` on sub-intrinsic/vega-collapse → `src/bs`.
5. **American / LSMC** → early-exercise premium; ITM-only regression, no look-ahead → `src/lsmc`.
6. **Variance reduction** → antithetic + control variate; 8.5× (control-alone beats combo, honestly) → `src/mc`.
7. **SVI + no-arb** → 5 params; w≥0 global-min + Lee `≤2` + Durrleman g → `src/surface/svi.py`.
8. **Inverse contracts** → premium in coin; USD = coin × index; both retained → `src/deribit/store.py`.
9. **Forward without a rate** → parity regression; `r = −ln(df)/τ` → `src/surface/forwards.py`.
10. **No-arb tests** → Durrleman g, price convexity, calendar monotonicity; 0 calendar viol → `src/noarb`.
11. **Exchange differential** → median \|Δσ\| 0.18/0.39; residuals peak at near-dated ATM → `src/exchdiff`.
12. **Convergence order** → CRR O(1/N), measured 0.9993, even-N ladder → `src/lattice/crr.py`.
13. **Term structure** → contango BTC 26→42%, ETH 34→56%; descriptive → `RESEARCH_NOTE §2`.
14. **Delta-hedge P&L** → long gamma / short theta; breakeven at realized = implied → `INTERVIEW_NOTES Q14`.
15. **Limitations** → descriptive only; 1-day window ⇒ IV-vs-realized deferred, not faked.
16. **Sticky-strike vs sticky-delta** → surface is log-moneyness (sticky-delta) parameterized; no dynamic claim.
17. **SVI vs SABR vs spline** → SVI fits total variance (matches calendar no-arb); SABR is dynamic; spline drops no-arb.
18. **Vanna/volga/skew-delta** → pieces present (dσ/dK + vega); RR ↔ vanna, BF ↔ volga.

## Tactics
- **Lead with the differentiator:** the *exchange differential* — our independent solver
  matching Deribit's own mark IV to a fraction of a vol point — is the thing no toy has.
- **When unsure / asked about edge:** *"It's descriptive; pricing is where I can be proven
  wrong, and I verified it three ways."* Pivot to cross-engine agreement.
- **On the data window (if pressed):** own it — *"two intraday snapshots, so I report
  cross-sectional shape and defer IV-vs-realized rather than fabricate a time series."*

## Pre-screen warm-up (what this project can't drill — do separately)
- [ ] Mental math — sub-8-second arithmetic (Optiver/IMC 80-in-8 style)
- [ ] Expected-value / fair-price games; market-make a number, update on a counterparty trade
- [ ] Combinatorics & probability — conditional/Bayes, binomial, gambler's ruin
- [ ] Sequential/strategy puzzles (backward induction — same logic as American exercise / LSMC)
