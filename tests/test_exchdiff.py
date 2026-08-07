"""Exchange differential tests: our solver's IV vs Deribit mark IV on the real fixture.

Descriptive-only per the mission (TOL["exch_diff_report_only"]): we PRINT the distribution
of delta_sigma = our_iv - mark_iv and assert only loose sanity bounds. A breach of the ATM
sanity bound below is a real finding to investigate, not a test to force-pass.
"""

from __future__ import annotations

import glob

import pytest

from src.deribit.store import load_snapshot
from src.exchdiff import ExchDiffResult, run_exchange_differential

# Loose sanity bound (NOT a registered pass/fail tolerance): liquid near-ATM options should
# agree with Deribit's mark IV to a few vol points. 0.05 = 5 vol points. If ATM median_abs
# exceeds this, that is a genuine finding to investigate, documented here and in the note.
_ATM_SANITY_BOUND = 0.05


def _latest_snapshot():
    path = sorted(glob.glob("data/snapshots/snapshot_*.json"))[-1]
    return load_snapshot(path)


@pytest.fixture(scope="module")
def snapshot():
    return _latest_snapshot()


@pytest.fixture(scope="module")
def results(snapshot):
    return {ccy: run_exchange_differential(snapshot, ccy) for ccy in ("BTC", "ETH")}


def _print_report(r: ExchDiffResult) -> None:
    print(f"\n=== exchange differential: {r.underlying} "
          f"(snapshot_ts={r.snapshot_ts:.0f}) ===")
    o = r.overall
    print(f"  matched N = {r.n_matched}")
    print(f"  overall   : median|d|={o.median_abs:.4f}  median={o.median:+.4f}  "
          f"IQR=[{o.q25:+.4f},{o.q75:+.4f}] mean={o.mean:+.4f} std={o.std:.4f} "
          "(vol points)")
    for b in ("ATM", "near", "wing"):
        s = r.by_moneyness[b]
        print(f"  moneyness {b:<5}: n={s.n:<4d} median|d|={s.median_abs:.4f} "
              f"median={s.median:+.4f}")
    for b in ("short", "medium", "long"):
        s = r.by_expiry[b]
        print(f"  expiry    {b:<6}: n={s.n:<4d} median|d|={s.median_abs:.4f} "
              f"median={s.median:+.4f}")


def test_matched_count_is_reasonable(results):
    """Both underlyings match a healthy number of (our_iv, mark_iv) pairs (> 100)."""
    for ccy, r in results.items():
        _print_report(r)
        assert r.n_matched > 100, f"{ccy}: only {r.n_matched} matched points"
        # bucket counts sum to the total (partition, no double-count / drop).
        assert sum(s.n for s in r.by_moneyness.values()) == r.n_matched
        assert sum(s.n for s in r.by_expiry.values()) == r.n_matched


def test_atm_agreement_is_small(results):
    """Liquid near-ATM median |delta_sigma| is small — a few vol points, under 5%.

    This is a sanity bound, not a registered tolerance. On the committed fixture BTC/ETH
    ATM median|d| ~ 0.002-0.004 (0.2-0.4 vol points); a breach would be investigated.
    """
    for ccy, r in results.items():
        atm = r.by_moneyness["ATM"]
        assert atm.n > 0, f"{ccy}: no ATM points"
        print(f"{ccy} ATM median|delta_sigma| = {atm.median_abs:.4f} vol points "
              f"(bound {_ATM_SANITY_BOUND})")
        assert atm.median_abs < _ATM_SANITY_BOUND, (
            f"{ccy}: ATM median|d|={atm.median_abs:.4f} exceeds sanity bound "
            f"{_ATM_SANITY_BOUND} — investigate (mark-IV construction / stale quotes), "
            "do not widen the bound to pass"
        )


def test_overall_distribution_is_sane(results):
    """Overall distribution is finite, tight, and roughly centered (no gross bias)."""
    import math

    for ccy, r in results.items():
        o = r.overall
        assert o.n == r.n_matched
        for v in (o.median, o.q25, o.q75, o.mean, o.std, o.median_abs):
            assert math.isfinite(v), f"{ccy}: non-finite stat {v}"
        assert o.q25 <= o.median <= o.q75
        assert o.iqr == pytest.approx(o.q75 - o.q25)
        assert o.median_abs >= 0.0
        # No systematic multi-vol-point bias between our solver and the exchange mark.
        assert abs(o.median) < _ATM_SANITY_BOUND, f"{ccy}: median bias {o.median:.4f}"


def test_outliers_populated_with_causes(results):
    """Outlier report is non-empty, sorted by descending |delta_sigma|, each has a cause."""
    for ccy, r in results.items():
        assert len(r.outliers) > 0, f"{ccy}: no outliers reported"
        abs_vals = [o.abs_delta_sigma for o in r.outliers]
        assert abs_vals == sorted(abs_vals, reverse=True), f"{ccy}: outliers not sorted"
        print(f"\n{ccy} top outliers:")
        for o in r.outliers[:5]:
            assert isinstance(o.cause, str) and len(o.cause) > 0
            assert o.abs_delta_sigma == pytest.approx(abs(o.point.delta_sigma))
            print(f"  |d|={o.abs_delta_sigma:.4f} {o.point.instrument_name} "
                  f"k={o.point.log_moneyness:+.2f} "
                  f"rel_spread={o.point.rel_spread} :: {o.cause}")
        # The single largest outlier is at least as large as the overall median_abs.
        assert r.outliers[0].abs_delta_sigma >= r.overall.median_abs


def test_determinism(snapshot):
    """Same fixture -> identical stats and identical outlier ordering."""
    for ccy in ("BTC", "ETH"):
        a = run_exchange_differential(snapshot, ccy)
        b = run_exchange_differential(snapshot, ccy)
        assert a.n_matched == b.n_matched
        assert a.overall == b.overall
        assert a.by_moneyness == b.by_moneyness
        assert a.by_expiry == b.by_expiry
        assert [o.point.instrument_name for o in a.outliers] == \
               [o.point.instrument_name for o in b.outliers]
        assert [o.abs_delta_sigma for o in a.outliers] == \
               [o.abs_delta_sigma for o in b.outliers]


def test_diagnosis_labels_match_observables(results):
    """Each diagnosed cause is consistent with the point's observable state."""
    for ccy, r in results.items():
        for o in r.outliers:
            p = o.point
            cause = o.cause
            if not p.used_mid:
                assert "mark-fallback" in cause, f"{ccy}: {p.instrument_name}"
            elif p.rel_spread is not None and p.rel_spread > 0.5:
                assert "wide spread" in cause, f"{ccy}: {p.instrument_name}"
            elif abs(p.log_moneyness) >= 0.20:
                assert "deep-OTM" in cause, f"{ccy}: {p.instrument_name}"
            else:
                assert "construction" in cause, f"{ccy}: {p.instrument_name}"


def test_no_live_api(monkeypatch):
    """Belt-and-suspenders: the differential path never touches the network."""
    import socket

    def _boom(*a, **k):  # pragma: no cover - only fires on a regression
        raise AssertionError("network access attempted in exchange differential")

    monkeypatch.setattr(socket, "socket", _boom)
    snap = _latest_snapshot()
    r = run_exchange_differential(snap, "BTC")
    assert r.n_matched > 100
