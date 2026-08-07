# vol-lab

[![CI](https://github.com/billdmar/vol-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/billdmar/vol-lab/actions/workflows/ci.yml)

**Options pricing and implied-volatility surface engine on real Deribit crypto-options data.**
Black-Scholes, CRR binomial, Monte Carlo, and Longstaff-Schwartz engines cross-verified
against each other, against theory, and against the exchange's own published mark IV — with
SVI surface calibration under no-arbitrage constraints and a strictly descriptive
BTC/ETH volatility research note.

> **Status: under construction.** This README is a Wave-0 stub; headline stats, figures,
> and the Verification section land as the build progresses (see `MISSION.md` in the
> private kit repo). Prime directive: right answers, proven — pricing has ground truth.

## What's here (and coming)
- `src/bs`, `src/lattice`, `src/mc`, `src/lsmc` — four pricing engines
- `src/deribit` — polite public-API snapshot collector (no auth, cached fixtures)
- `src/surface` — parity-inferred forwards, per-expiry smiles, SVI calibration
- `src/noarb`, `src/exchdiff` — no-arbitrage scan + exchange differential vs mark IV
- `docs/RESEARCH_NOTE.md` — descriptive vol study with confidence intervals, zero forecasts

## Quickstart
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy==2.5.1 scipy==1.18.0 pandas==3.0.5 matplotlib==3.11.1 \
            hypothesis==6.165.2 pytest==9.1.1 coverage==7.15.4 ruff==0.16.1 requests==2.34.2
ruff check . && coverage run -m pytest && coverage report
```

## License
MIT — see [LICENSE](LICENSE).
