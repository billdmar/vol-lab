# RESUME_BULLETS.md — vol-lab (filled with measured values)

Three bullets in the mission's §7 format, populated with the project's real figures.
Repo: https://github.com/billdmar/vol-lab

- Built an options pricing and implied-volatility platform (Black-Scholes, CRR binomial,
  Monte Carlo, Longstaff-Schwartz) on **live Deribit market data**, calibrating SVI smiles
  across **1,540 instruments (BTC + ETH) over 12 expiries per underlying** under
  no-arbitrage constraints (positive-variance + Lee wing-slope bounds, butterfly and
  calendar scans).

- Verified numerics by **cross-engine differentials with a measured binomial→Black-Scholes
  convergence order of 0.9993**, three-way Greeks reconciliation (closed-form vs finite
  differences vs Monte-Carlo pathwise/likelihood-ratio, agreeing to `<0.5%`), put-call
  parity to machine precision (~2e-14), and an **exchange differential matching Deribit's
  published mark IV to a median |Δσ| of 0.18 vol points (BTC) / 0.39 (ETH)** — with
  **96% test coverage** and green CI on committed fixtures.

- Authored a **descriptive BTC/ETH volatility study** (contango term structure, 25-delta
  risk-reversal and butterfly, smile skew — including ETH's front-end skew *flip*) with
  **confidence intervals and zero forecasts or trading claims**, plus a measured Monte-Carlo
  variance-reduction speedup of **8.5× (control variate)**.

---

## Shorter single-line variant (for a dense resume)

> **vol-lab** — Options pricing + IV-surface engine on live Deribit data: 4 cross-verified
> engines (BS/binomial/MC/LSMC), SVI calibration under no-arb constraints, IVs matching the
> exchange's mark IV to a median 0.18–0.39 vol points; measured convergence order 0.9993,
> 8.5× MC variance-reduction, 96% coverage, CI-green. Python/NumPy/SciPy. `github.com/billdmar/vol-lab`

## Talking points (for the screen itself)

- *"Pricing has right answers, so I built the project where I could be proven wrong."*
- Parity used operationally (forward inference), not just recited.
- Greeks implemented three independent ways and reconciled — can walk each on a whiteboard.
- The exchange differential is the differentiator: an *external* check no textbook toy has.
- Honest about the window: the time-series (IV vs realized) analysis is deferred until more
  snapshot days accumulate, rather than fabricated on one day of data.

*Pair with mental-math and probability drilling — the options-MM screens weight those
alongside the theory this project demonstrates (see `docs/INTERVIEW_NOTES.md`).*
