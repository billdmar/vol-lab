#!/usr/bin/env python3
"""Reproducible surface/verification report (ORCH, G2) — one command, every statistic.

Runs the full surface pipeline on the committed snapshot fixtures and prints the
descriptive statistics the research note and README cite: per-expiry SVI RMSE, ATM
term structure, 25-delta RR/BF, the no-arbitrage scan, and the exchange differential
(our IV vs Deribit mark IV). Deterministic: same fixtures -> identical output.

Usage:
    python scripts/report_surface.py                 # latest snapshot, BTC+ETH
    python scripts/report_surface.py --all-snapshots # every committed snapshot (stability)
    python scripts/report_surface.py --json          # machine-readable dump
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

# Run-as-script: put the repo root on sys.path so `src` / `config` import (pytest uses
# pyproject's pythonpath, but a direct `python scripts/...` invocation does not).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deribit.store import load_snapshot  # noqa: E402
from src.exchdiff.differential import run_exchange_differential  # noqa: E402
from src.noarb.scan import scan_surface  # noqa: E402
from src.surface import build_surface  # noqa: E402


def _expiry_label(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%d%b%y")


def report_snapshot(path: str, underlyings=("BTC", "ETH")) -> dict:
    snap = load_snapshot(path)
    collected_iso = dt.datetime.fromtimestamp(snap.collected_ts, dt.UTC).isoformat()
    out: dict = {"snapshot": path.split("/")[-1], "collected_iso": collected_iso,
                 "index_prices": snap.index_prices, "underlyings": {}}
    print(f"\n{'=' * 78}\nSNAPSHOT {out['snapshot']}  ({collected_iso})")
    print(f"index {snap.index_prices}")
    for ccy in underlyings:
        quotes = snap.for_underlying(ccy)
        surf = build_surface(quotes, ref_ts=snap.collected_ts)
        scan = scan_surface(surf)
        exch = run_exchange_differential(snap, ccy)

        print(f"\n--- {ccy}: {len(surf.slices)} calibrated slices "
              f"(of {len(snap.expiries(ccy))} expiries) ---")
        print(f"{'expiry':>9} {'tau':>6} {'F':>9} {'dF%':>6} {'npts':>4} "
              f"{'ATMvol':>7} {'rmse_w':>8} {'RR25':>7} {'BF25':>6}")
        slices_out = []
        for s in surf.slices:
            dfpct = (100 * (s.forward - s.fit.deribit_forward) / s.fit.deribit_forward
                     if s.fit.deribit_forward else float("nan"))
            rr = f"{s.rr_25 * 100:+.2f}" if s.rr_25 is not None else "   n/a"
            bf = f"{s.bf_25 * 100:.2f}" if s.bf_25 is not None else " n/a"
            print(f"{_expiry_label(s.expiry_ts):>9} {s.tau:>6.3f} {s.forward:>9.0f} "
                  f"{dfpct:>+6.2f} {len(s.smile.points):>4} {s.atm_vol * 100:>6.1f}% "
                  f"{s.svi.rmse_w:>8.1e} {rr:>7} {bf:>6}")
            slices_out.append({
                "expiry": _expiry_label(s.expiry_ts), "tau": s.tau, "forward": s.forward,
                "deribit_forward": s.fit.deribit_forward, "n_points": len(s.smile.points),
                "atm_vol": s.atm_vol, "rmse_w": s.svi.rmse_w, "rmse_vol": s.svi.rmse_vol,
                "rr_25": s.rr_25, "bf_25": s.bf_25,
                "svi": s.svi.params.as_dict(),
            })

        worst_pb = (scan.worst_price_butterfly.magnitude
                    if scan.worst_price_butterfly is not None else 0.0)
        gmin = scan.worst_g_min if scan.worst_g_min is not None else 0.0
        print(f"\n  no-arb scan: SVI butterfly g<0 = {scan.n_slice_butterfly_violations} "
              f"(worst g_min={gmin:.2e} @ {scan.worst_g_location}); "
              f"price-convexity violations = {scan.n_price_butterfly_violations} "
              f"(worst={worst_pb:.4f} USD); "
              f"calendar violations = {scan.n_calendar_violations}")

        o = exch.overall
        print(f"  exchange differential (our IV vs Deribit mark IV): "
              f"{exch.n_matched} matched; "
              f"median|d-sigma|={o.median_abs * 100:.3f} vol pts, "
              f"median={o.median * 100:+.3f}, IQR={o.iqr * 100:.3f}")
        for bname, st in exch.by_moneyness.items():
            print(f"       moneyness[{bname}]: n={st.n} "
                  f"median|d-sigma|={st.median_abs * 100:.3f} vol pts")

        out["underlyings"][ccy] = {
            "n_slices": len(surf.slices),
            "slices": slices_out,
            "noarb": {
                "n_slice_butterfly": scan.n_slice_butterfly_violations,
                "worst_g_min": scan.worst_g_min,
                "worst_g_location": scan.worst_g_location,
                "n_price_butterfly": scan.n_price_butterfly_violations,
                "worst_price_butterfly": worst_pb,
                "n_calendar": scan.n_calendar_violations,
            },
            "exchdiff": {
                "n_matched": exch.n_matched,
                "overall": {"median_abs": o.median_abs, "median": o.median, "iqr": o.iqr,
                            "mean": o.mean, "std": o.std},
                "by_moneyness": {b: {"n": s.n, "median_abs": s.median_abs}
                                 for b, s in exch.by_moneyness.items()},
                "by_expiry": {b: {"n": s.n, "median_abs": s.median_abs}
                              for b, s in exch.by_expiry.items()},
                "top_outliers": [{"instrument": ol.point.instrument_name,
                                  "abs_delta_sigma": ol.abs_delta_sigma, "cause": ol.cause}
                                 for ol in exch.outliers[:5]],
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproducible vol-lab surface/verification report.")
    ap.add_argument("--all-snapshots", action="store_true", help="report every committed snapshot")
    ap.add_argument("--json", action="store_true", help="also print a machine-readable JSON dump")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob("data/snapshots/snapshot_*.json"))
    if not paths:
        print("no snapshots found under data/snapshots/")
        return 1
    if not args.all_snapshots:
        paths = paths[-1:]

    reports = [report_snapshot(p) for p in paths]
    if args.json:
        print("\n" + json.dumps(reports, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
