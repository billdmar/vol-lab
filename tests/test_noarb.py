"""Tests for the static no-arbitrage scanner (src/noarb).

Coverage:
  * Analytic SVI derivatives (w', w'') and Durrleman g(k) agree with tight central finite
    differences — validates the g formula independently of the analytic algebra.
  * A deliberately arbitrageable SVI slice (large b, sharp wing) is FLAGGED by the butterfly
    scan; a calm arb-free slice passes.
  * A synthetic calendar violation (inner-tau variance pushed above outer-tau) is flagged;
    a properly increasing term structure is not.
  * The scanner RUNS on the real fixture surface for BTC and ETH and returns sane structured
    output; violation counts + worst magnitudes are PRINTED (descriptive, not asserted zero).
"""

from __future__ import annotations

import glob

import numpy as np
import pytest

from config.tolerances import TOL
from src.deribit.store import load_snapshot
from src.noarb import (
    durrleman_g,
    scan_calendar,
    scan_slice_butterfly,
    scan_surface,
    svi_w_derivatives,
)
from src.noarb.scan import scan_price_butterfly
from src.surface import SVIParams, build_surface, svi_total_variance

# A calm, arb-free slice: modest wings, gentle curvature (typical liquid BTC smile).
_CALM = SVIParams(a=0.02, b=0.10, rho=-0.20, m=0.0, sigma=0.20)


# --------------------------------------------------------------------------- derivatives


def test_svi_derivatives_match_finite_difference():
    """Analytic w', w'' match tight central finite differences of w(k)."""
    ks = np.linspace(-1.0, 1.0, 21)
    h = 1e-5
    _, wp, wpp = svi_w_derivatives(ks, _CALM)
    wp_fd = (svi_total_variance(ks + h, _CALM) - svi_total_variance(ks - h, _CALM)) / (2 * h)
    wpp_fd = (
        svi_total_variance(ks + h, _CALM)
        - 2 * svi_total_variance(ks, _CALM)
        + svi_total_variance(ks - h, _CALM)
    ) / (h * h)
    assert np.allclose(wp, wp_fd, atol=1e-6, rtol=1e-6)
    assert np.allclose(wpp, wpp_fd, atol=1e-4, rtol=1e-4)


def test_durrleman_g_matches_finite_difference():
    """g(k) built from analytic derivatives equals g(k) built from FD derivatives."""
    ks = np.linspace(-0.8, 0.8, 33)
    h = 1e-5

    def g_fd(k: np.ndarray) -> np.ndarray:
        w = svi_total_variance(k, _CALM)
        wp = (svi_total_variance(k + h, _CALM) - svi_total_variance(k - h, _CALM)) / (2 * h)
        wpp = (
            svi_total_variance(k + h, _CALM)
            - 2 * svi_total_variance(k, _CALM)
            + svi_total_variance(k - h, _CALM)
        ) / (h * h)
        return (1.0 - k * wp / (2 * w)) ** 2 - (wp * wp / 4) * (1 / w + 0.25) + wpp / 2

    assert np.allclose(durrleman_g(ks, _CALM), g_fd(ks), atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- SVI butterfly


def test_arb_free_slice_passes_butterfly():
    """A calm SVI slice has g(k) >= 0 everywhere and is not flagged."""
    res = scan_slice_butterfly(_CALM, tau=0.25, expiry_ts=0.0)
    assert not res.violated
    assert res.g_min >= TOL["svi_butterfly_g_min"].value


def test_arbitrageable_slice_is_flagged():
    """A slice with a huge wing slope (large b) violates the butterfly condition."""
    # Large b with tight sigma makes w very steep near m -> g(k) dips negative on a wing.
    bad = SVIParams(a=0.005, b=1.2, rho=-0.85, m=0.0, sigma=0.02)
    res = scan_slice_butterfly(bad, tau=0.25, expiry_ts=0.0)
    assert res.violated
    assert res.g_min < TOL["svi_butterfly_g_min"].value
    # The reported location is a real grid point in the scanned range.
    assert -1.5 <= res.k_at_min <= 1.5


# --------------------------------------------------------------------------- calendar


class _FakeSmile:
    def __init__(self, ks):
        self._k = np.asarray(ks, dtype=float)

    @property
    def log_moneyness(self):
        return self._k


class _FakeSlice:
    def __init__(self, tau, expiry_ts, params, ks):
        self.tau = tau
        self.expiry_ts = expiry_ts

        class _S:
            pass

        self.svi = _S()
        self.svi.params = params
        self.smile = _FakeSmile(ks)


class _FakeSurface:
    def __init__(self, slices):
        self.underlying = "TEST"
        self.slices = tuple(slices)


def test_calendar_violation_is_flagged():
    """Inner-tau total variance above outer-tau at fixed k is flagged as a crossing."""
    ks = np.linspace(-0.5, 0.5, 11)
    # inner (short tau) has HIGHER variance level than outer (long tau) -> calendar arb.
    inner = SVIParams(a=0.10, b=0.05, rho=0.0, m=0.0, sigma=0.20)
    outer = SVIParams(a=0.04, b=0.05, rho=0.0, m=0.0, sigma=0.20)
    surf = _FakeSurface([
        _FakeSlice(0.10, 1.0, inner, ks),
        _FakeSlice(0.50, 2.0, outer, ks),
    ])
    viols = scan_calendar(surf, n_grid=21)
    assert len(viols) > 0
    v = max(viols, key=lambda x: x.gap)
    assert v.gap > 0.0
    assert v.tau_inner < v.tau_outer
    assert v.w_inner > v.w_outer


def test_calendar_ok_when_variance_increases():
    """A properly increasing total-variance term structure yields no calendar violation."""
    ks = np.linspace(-0.5, 0.5, 11)
    inner = SVIParams(a=0.02, b=0.05, rho=0.0, m=0.0, sigma=0.20)
    outer = SVIParams(a=0.10, b=0.05, rho=0.0, m=0.0, sigma=0.20)
    surf = _FakeSurface([
        _FakeSlice(0.10, 1.0, inner, ks),
        _FakeSlice(0.50, 2.0, outer, ks),
    ])
    assert scan_calendar(surf, n_grid=21) == []


# --------------------------------------------------------------------------- price butterfly


def test_price_butterfly_flags_nonconvex_calls():
    """A non-convex call-price triplet is flagged with the three strikes and magnitude."""
    class _P:
        def __init__(self, strike):
            self.strike = strike

    class _Smile:
        expiry_ts = 0.0
        tau = 0.25
        points = (_P(90.0), _P(100.0), _P(110.0))

    # Concave (arb): middle strike too expensive -> right slope < left slope.
    # left=(11-12)/10=-0.1, right=(2-11)/10=-0.9 -> convexity=-0.8 < 0.
    prices = {90.0: 12.0, 100.0: 11.0, 110.0: 2.0}
    viols = scan_price_butterfly(_Smile(), call_prices=prices)
    assert len(viols) == 1
    assert viols[0].strikes == (90.0, 100.0, 110.0)
    assert viols[0].magnitude > 0.0


def test_price_butterfly_passes_convex_calls():
    """A convex, decreasing call curve produces no butterfly violation."""
    class _P:
        def __init__(self, strike):
            self.strike = strike

    class _Smile:
        expiry_ts = 0.0
        tau = 0.25
        points = (_P(90.0), _P(100.0), _P(110.0))

    prices = {90.0: 12.0, 100.0: 5.0, 110.0: 2.0}  # convex, monotone decreasing
    assert scan_price_butterfly(_Smile(), call_prices=prices) == []


# --------------------------------------------------------------------------- real surface


@pytest.fixture(scope="module")
def snapshot():
    path = sorted(glob.glob("data/snapshots/snapshot_*.json"))[-1]
    return load_snapshot(path)


@pytest.mark.parametrize("ccy", ["BTC", "ETH"])
def test_scanner_runs_on_real_surface(snapshot, ccy):
    """Scanner runs on the real fixture surface and returns sane structured output.

    Descriptive: real markets carry some arb noise, so counts are REPORTED, not asserted
    zero. We assert only that the scan runs and produces internally-consistent structure.
    """
    quotes = snapshot.for_underlying(ccy)
    surface = build_surface(quotes, ref_ts=snapshot.collected_ts)
    assert len(surface.slices) > 0
    res = scan_surface(surface)

    # Structural sanity: counts match the detail lists.
    assert res.underlying == ccy
    assert res.n_slice_butterfly_violations == sum(1 for r in res.slice_butterfly if r.violated)
    assert res.n_price_butterfly_violations == len(res.price_butterfly)
    assert res.n_calendar_violations == len(res.calendar)
    assert res.total_violations >= 0
    assert len(res.slice_butterfly) == len(surface.slices)

    worst_g = f"{res.worst_g_min:.3e}" if res.worst_g_min is not None else "n/a"
    worst_pb = f"{res.worst_price_butterfly.magnitude:.4g}" if res.worst_price_butterfly else "n/a"
    worst_cal = f"{res.worst_calendar_gap:.4g}" if res.worst_calendar_gap is not None else "n/a"
    print(
        f"\n[{ccy}] slices={len(surface.slices)} "
        f"butterfly(SVI g<0)={res.n_slice_butterfly_violations} (worst g_min={worst_g}) "
        f"butterfly(price)={res.n_price_butterfly_violations} (worst={worst_pb} USD/K) "
        f"calendar={res.n_calendar_violations} (worst gap={worst_cal} tot-var)"
    )
