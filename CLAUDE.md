# vol-lab — standing instructions for Claude Code

## What this is
An options pricing and implied-volatility surface engine built on real Deribit
crypto-options market data: Black-Scholes + binomial + Monte Carlo + LSMC
engines cross-verified against each other and against the exchange's own
published mark IV, with SVI surface calibration under no-arbitrage constraints
and a descriptive volatility research note. The owner uses it as the
derivatives/quant-research complement to nanobook (systems) for Summer 2027
quant internships at options market makers (Akuna, CTC, Optiver, SIG, IMC).
Verified numerics and honest claims matter more than feature count.

## Mission mode
The complete autonomous build plan lives in MISSION.md (repo root — gitignored
in the project repo, backed up to the private billdmar/vol-lab-kit repo).
This kit is fully self-contained: you have NO access to any prior conversation
or external memory; everything you need is in these files. At the start of ANY
session: read MISSION.md, the tracker below, and `git log --oneline -15`, then
continue the mission without being asked.

## Autonomy protocol
- Execute wave by wave (MISSION.md §5); a gate finishes only when its
  Definition of Done passes with fresh command output as evidence.
- Interrupt the human ONLY for: passwords/sudo, browser logins, GUI installs,
  git identity (once), or a destructive/ambiguous choice. There is no required
  product decision — underlyings default to BTC + ETH. Decide the rest; log it.
- Every gate: tick tracker, commit, push BOTH repos, print DoD evidence +
  headline stats + what's next. Long context: close the wave, commit, tell the
  human to /clear and resume.

## Parallel execution protocol
- Subagents + background bash per MISSION.md; monitor with /tasks.
- Contracts freeze at W0 (market-snapshot schema, pricer/Greeks interfaces,
  surface object, tolerance registry); changes go through the orchestrator.
- One writer per path (ownership map in MISSION.md); conflict = escalate.
- Permission warm-up before any background fan-out (pip/python, pytest, git,
  curl): background subagents auto-deny new permission prompts.
- Integration gates are sequential, never parallel: cross-engine
  differentials, parity, no-arb scans, exchange differential, coverage.
- Background jobs: Deribit snapshot collection (rate-limited), long MC runs,
  property-test sweeps.

## Hard rules
- Dev machine: Apple Silicon Mac. CI: ubuntu-latest. Python 3.12 in a project
  venv; deps pinned: numpy, scipy, pandas, matplotlib, hypothesis, pytest,
  coverage, ruff, requests. No heavy frameworks; no paid data ever.
- **Deribit etiquette:** public endpoints only, no auth, descriptive
  User-Agent, ≥ 250ms spacing between requests, cache every response under
  data/snapshots/ (committed as fixtures), and CI NEVER calls the live API.
- **Inverse contracts handled explicitly:** premiums are quoted in coin;
  convert via the index price and document the convention in DESIGN.md.
  Forwards are inferred from put-call parity on liquid strikes, not assumed.
- **Tolerance registry:** every differential test's tolerance lives in one
  file (config/tolerances.py) with a written justification per entry.
  Tolerances are never widened to make a failing test pass — a failure means
  investigate, and the investigation gets written down.
- Seeded determinism everywhere randomness exists. Every published statistic
  reproducible by one documented command.
- **No trading claims:** no PnL, no strategy, no "alpha," no forecasts sold
  as facts. The research note is descriptive, with confidence intervals.
- Every modeling choice (day count, rate assumption, filtering of illiquid
  strikes, SVI constraint set) gets 2–4 lines of rationale in docs/DESIGN.md.

## Definition of done — every gate, no exceptions
1. `pytest` all green; `ruff check` clean; hypothesis property suites pass.
2. Cross-engine differentials green: binomial→BS convergence with measured
   order; MC confidence intervals cover closed form; LSMC vs binomial
   (American) within registered tolerance.
3. Greeks differential green: closed form vs central finite differences vs MC
   pathwise, per the tolerance registry.
4. Put-call parity: machine precision on model prices; documented bounded
   residuals on market snapshots.
5. From G2 on: no-arbitrage scan (butterfly ≥ 0, calendar ≥ 0) run on every
   calibrated surface with violations quantified; SVI fit RMSE reported;
   exchange differential vs Deribit mark IV reported with outliers
   investigated in writing.
6. Coverage ≥ 90% on src/ engines — report the number. Determinism holds.
7. Docs current (DESIGN/RESEARCH_NOTE/README). Committed and pushed. Never
   commit MISSION.md/START-HERE.md/build-kit to the PROJECT repo (they live
   in vol-lab-kit); snapshot fixtures ARE committed.

## Never do
- Never widen a tolerance, drop a test case, or smooth an arbitrage violation
  to turn red green — investigate and document instead.
- Never claim agreement with the exchange without printing the distribution
  of differences. Never cherry-pick snapshot days.
- Never present the research note as tradeable insight or forecast.
- Never exceed the request-spacing rule or call live Deribit from CI.
- Never let a subagent edit outside its owned paths or alter frozen contracts.

## Tracker (tick when the gate's DoD fully passes)
- [x] W0 bootstrap + scaffold + CI + contracts frozen + snapshot collection
      started + private kit repo (billdmar/vol-lab-kit) created and pushed
- [x] G1 engines merged and cross-verified: BS, binomial, MC, Greeks,
      property suites — convergence orders measured (CRR->BS order 0.9993;
      MC CI coverage 6/6; Greeks 3-way rel <0.5%; parity 4.3e-14; cov 98%)
- [ ] G2 surfaces: parity-inferred forwards, per-expiry smiles, SVI
      calibration, no-arb scan, exchange differential vs mark IV
- [ ] G3 research note written from N snapshot days; all figures generated;
      LSMC American pricing verified
- [ ] G4 recruiter-grade README + INTERVIEW_NOTES + RESUME_BULLETS — complete
- [ ] P6 (stretch, only if asked) pybind11 C++ MC kernel / CBOE delayed
      equity-options extension
