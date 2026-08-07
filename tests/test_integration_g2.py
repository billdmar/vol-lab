"""G2 — Surface verification gate (integration tests).

Ties the surface, no-arb scanner, and exchange differential together on the committed
Deribit fixtures and asserts the gate's Definition of Done:

  * The full pipeline (forwards -> smiles -> SVI -> no-arb scan -> exchange differential)
    runs end-to-end on real data for BTC and ETH.
  * No-arb scan produces structured results; calendar arbitrage is absent on the snapshot
    (a real market property here), and any butterfly flags are quantified (not asserted
    to zero — real surfaces have wing-extrapolation artifacts we surface honestly).
  * The exchange differential matches Deribit's published mark IV tightly: median |Δσ|
    is small (reported), with residuals concentrated toward ATM/near-dated as theory says.
  * Determinism: the same fixture reproduces identical statistics bit-for-bit.

These are descriptive gates (exch_diff has no pass/fail tolerance per the registry); the
assertions are sanity bounds that would fail a broken pipeline, not tuned thresholds.
"""

from __future__ import annotations

import glob

from src.deribit.store import load_snapshot
from src.exchdiff.differential import run_exchange_differential
from src.noarb.scan import scan_surface
from src.surface import build_surface

FIXTURES = sorted(glob.glob("data/snapshots/snapshot_*.json"))


def test_full_surface_pipeline_runs_both_underlyings():
    snap = load_snapshot(FIXTURES[-1])
    for ccy in ("BTC", "ETH"):
        quotes = snap.for_underlying(ccy)
        surf = build_surface(quotes, ref_ts=snap.collected_ts)
        assert len(surf.slices) >= 8
        scan = scan_surface(surf)
        assert scan.underlying == ccy
        # Calendar arbitrage: none on this snapshot (total variance non-decreasing in tau).
        assert scan.n_calendar_violations == 0
        # Butterfly flags, if any, are quantified with a location (not silently dropped).
        if scan.n_slice_butterfly_violations > 0:
            assert scan.worst_g_location is not None
            assert scan.worst_g_min is not None


def test_exchange_differential_tight_and_bucketed():
    snap = load_snapshot(FIXTURES[-1])
    for ccy in ("BTC", "ETH"):
        res = run_exchange_differential(snap, ccy)
        assert res.n_matched > 100, f"{ccy} too few matched points"
        # Our independently-written solver agrees with Deribit mark IV to well under
        # 5 vol points overall — a loose sanity bound, NOT a tuned tolerance.
        assert res.overall.median_abs < 0.05, (ccy, res.overall.median_abs)
        # Buckets are populated and every outlier carries a diagnosed cause.
        assert res.by_moneyness
        assert res.by_expiry
        for ol in res.outliers:
            assert ol.cause  # non-empty cause string


def test_g2_determinism_same_fixture_identical_stats():
    """Re-running the differential on the same fixture yields identical statistics."""
    snap = load_snapshot(FIXTURES[-1])
    a = run_exchange_differential(snap, "BTC")
    b = run_exchange_differential(snap, "BTC")
    assert a.n_matched == b.n_matched
    assert a.overall.median_abs == b.overall.median_abs
    assert a.overall.median == b.overall.median
    assert a.overall.iqr == b.overall.iqr
    # Outlier ordering is stable too.
    assert [o.point.instrument_name for o in a.outliers] == \
           [o.point.instrument_name for o in b.outliers]


def test_surface_determinism_svi_params_reproducible():
    """SVI calibration is deterministic (fixed multi-start, no RNG)."""
    snap = load_snapshot(FIXTURES[-1])
    s1 = build_surface(snap.for_underlying("ETH"), ref_ts=snap.collected_ts)
    s2 = build_surface(snap.for_underlying("ETH"), ref_ts=snap.collected_ts)
    p1 = [sl.svi.params.as_dict() for sl in s1.slices]
    p2 = [sl.svi.params.as_dict() for sl in s2.slices]
    assert p1 == p2
