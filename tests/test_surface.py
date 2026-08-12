"""Surface construction tests: forwards, smiles, SVI.

Two layers of verification:
  * Synthetic exact-recovery: build option prices from a KNOWN forward and a KNOWN SVI
    slice, then confirm we recover the forward (parity regression) and the SVI shape (fit).
    This is the "pricing has right answers" discipline applied to the surface.
  * Real-data sanity: run the full pipeline on the committed Deribit fixture and assert the
    structural properties (forwards match Deribit's own forward, RMSE reported, RR/BF finite).
"""

from __future__ import annotations

import glob
import math

import numpy as np
import pytest

from config.tolerances import TOL
from src.bs import BS
from src.deribit.store import load_snapshot
from src.schema import OptionQuote
from src.surface import (
    SVIParams,
    build_smile,
    build_surface,
    calibrate_svi,
    infer_forward,
    infer_forwards_all_expiries,
    svi_iv,
    svi_total_variance,
    year_fraction,
)

_SNAPSHOTS = sorted(glob.glob("data/snapshots/snapshot_*.json"))
FIXTURE = _SNAPSHOTS[-1]


# ------------------------------------------------------- synthetic exact recovery


def _synthetic_quotes(forward, tau, rate, sigma, strikes, underlying="BTC",
                      index=60000.0, snapshot_ts=1_749_000_000.0, expiry_ts=1_750_000_000.0):
    """Build call+put OptionQuotes priced at a flat BS sigma on a known forward.

    Premiums are stored back in coin units (USD price / index) so the parity regression
    and IV solver see a self-consistent inverse-contract quote.
    """
    spot = forward * math.exp(-rate * tau)
    quotes = []
    for k in strikes:
        for ot in ("C", "P"):
            usd = BS.price(spot=spot, strike=k, tau=tau, rate=rate, sigma=sigma,
                           option_type=ot, carry=0.0).price
            coin = usd / index
            quotes.append(OptionQuote(
                instrument_name=f"{underlying}-X-{int(k)}-{ot}",
                underlying=underlying, option_type=ot, strike=float(k),
                expiry_ts=expiry_ts, bid_coin=coin * 0.999, ask_coin=coin * 1.001,
                mark_price_coin=coin, mark_price_usd=usd, mark_iv=sigma,
                open_interest=100.0, index_price=index, underlying_price=forward,
                snapshot_ts=snapshot_ts,
            ))
    return quotes


def test_forward_exact_recovery_from_parity():
    """A flat-vol synthetic book recovers its known forward to high precision."""
    F, tau, rate, sigma = 61234.0, 0.25, 0.0, 0.6
    strikes = np.linspace(50000, 75000, 15)
    ref_ts = 1_750_000_000.0 - tau * 365 * 86400
    quotes = _synthetic_quotes(F, tau, rate, sigma, strikes, snapshot_ts=ref_ts)
    fit = infer_forward(quotes, ref_ts=ref_ts)
    assert fit is not None
    assert fit.forward == pytest.approx(F, rel=1e-6)
    assert fit.discount_factor == pytest.approx(1.0, abs=1e-6)  # rate 0 -> df 1
    assert fit.n_pairs == len(strikes)
    assert fit.resid_rms < 1e-3


def test_forward_recovery_with_nonzero_rate():
    F, tau, rate, sigma = 61000.0, 0.5, 0.05, 0.5
    strikes = np.linspace(50000, 72000, 12)
    ref_ts = 1_750_000_000.0 - tau * 365 * 86400
    quotes = _synthetic_quotes(F, tau, rate, sigma, strikes, snapshot_ts=ref_ts)
    fit = infer_forward(quotes, ref_ts=ref_ts)
    assert fit is not None
    assert fit.forward == pytest.approx(F, rel=1e-5)
    assert fit.discount_factor == pytest.approx(math.exp(-rate * tau), abs=1e-5)
    assert fit.rate == pytest.approx(rate, abs=1e-4)


def test_forward_none_when_too_few_pairs():
    F, tau, sigma = 60000.0, 0.25, 0.6
    ref_ts = 1_750_000_000.0 - tau * 365 * 86400
    quotes = _synthetic_quotes(F, tau, 0.0, sigma, [60000.0], snapshot_ts=ref_ts)
    assert infer_forward(quotes, ref_ts=ref_ts, min_pairs=3) is None


def test_smile_flat_vol_recovers_sigma():
    """A flat-vol book yields a flat smile at the known sigma (IV solver round-trip)."""
    F, tau, sigma = 60000.0, 0.25, 0.55
    strikes = np.linspace(48000, 74000, 15)
    ref_ts = 1_750_000_000.0 - tau * 365 * 86400
    quotes = _synthetic_quotes(F, tau, 0.0, sigma, strikes, snapshot_ts=ref_ts)
    fit = infer_forward(quotes, ref_ts=ref_ts)
    smile = build_smile(quotes, fit)
    assert len(smile.points) >= 8
    assert smile.ivs == pytest.approx(np.full(len(smile.points), sigma), abs=1e-4)
    # OTM selection: calls above F, puts below F.
    for p in smile.points:
        if p.option_type == "C":
            assert p.strike >= F
        else:
            assert p.strike < F


def test_svi_recovers_known_slice():
    """Prices generated from a known SVI slice re-calibrate to (near) the same params."""
    true = SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.15)
    tau = 0.5
    k = np.linspace(-0.6, 0.6, 21)
    w = svi_total_variance(k, true)
    fit = calibrate_svi(k, w, tau)
    # Recovered variance curve matches the true one very tightly (params can trade off,
    # so we assert on the CURVE, which is what actually matters for the surface).
    w_fit = svi_total_variance(k, fit.params)
    assert np.allclose(w_fit, w, atol=1e-4)
    assert fit.rmse_w < 1e-3
    assert fit.converged


def test_svi_needs_enough_points():
    with pytest.raises(ValueError, match=">= 5 points"):
        calibrate_svi([0.0, 0.1, 0.2], [0.04, 0.041, 0.045], 0.5)


def test_svi_iv_nonnegative():
    params = SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.15)
    ivs = svi_iv(np.linspace(-1, 1, 11), params, 0.5)
    assert np.all(ivs >= 0.0)


def test_year_fraction_act365():
    assert year_fraction(1000.0 + 365 * 86400, 1000.0) == pytest.approx(1.0)
    assert year_fraction(500.0, 1000.0) == 0.0  # past expiry clamps to 0


# ----------------------------------------------------------- real-data pipeline


def test_build_surface_real_fixture_btc_and_eth():
    """Full pipeline on the committed Deribit fixture: structural sanity + external check."""
    snap = load_snapshot(FIXTURE)
    for ccy in ("BTC", "ETH"):
        quotes = snap.for_underlying(ccy)
        surf = build_surface(quotes, ref_ts=snap.collected_ts)
        assert len(surf.slices) >= 8, f"{ccy} too few calibrated slices"
        for s in surf.slices:
            # Parity forward matches Deribit's own forward to < 1% (external validation).
            if s.fit.deribit_forward is not None:
                rel = abs(s.forward - s.fit.deribit_forward) / s.fit.deribit_forward
                assert rel < 0.01, f"{ccy} {s.expiry_ts}: forward off by {rel:.3%}"
            # ATM vol is a sane crypto level.
            assert 0.05 < s.atm_vol < 3.0
            # SVI fit produced a finite RMSE and the slice is calibrated.
            assert math.isfinite(s.svi.rmse_w)
            assert s.svi.n_points >= 5
        # Term structure is monotone-ish (sorted by tau, all finite).
        ts = surf.term_structure
        assert all(math.isfinite(v) for _, v in ts)


def test_build_surface_empty_and_min_points_guards():
    """Fail-loud on no quotes; skip expiries with too few smile points (honest-unknown)."""
    with pytest.raises(ValueError, match="no quotes"):
        build_surface([], ref_ts=0.0)

    snap = load_snapshot(FIXTURE)
    quotes = snap.for_underlying("BTC")
    # An unreachable min forces every expiry to be skipped -> zero slices, no crash.
    surf = build_surface(quotes, ref_ts=snap.collected_ts, min_smile_points=10_000)
    assert surf.slices == ()


def test_forward_negative_df_returns_none():
    """A book whose C-P slopes upward in K (nonsensical) yields no forward, not a fake one."""
    # Construct pairs where C - P increases with K (positive slope -> df<0).
    from src.schema import OptionQuote as OQ
    ref_ts = 0.0
    quotes = []
    for i, k in enumerate([100.0, 110.0, 120.0, 130.0]):
        cu, pu = 1.0 + i, 0.0  # C-P = 1+i rises with K -> slope>0 -> df<0
        for ot, coin in (("C", cu / 100.0), ("P", pu / 100.0)):
            quotes.append(OQ(
                instrument_name=f"BTC-X-{int(k)}-{ot}", underlying="BTC", option_type=ot,
                strike=k, expiry_ts=1_750_000_000.0, bid_coin=None, ask_coin=None,
                mark_price_coin=max(coin, 1e-9), mark_price_usd=max(coin, 1e-9) * 100.0,
                mark_iv=0.5, open_interest=1.0, index_price=100.0, underlying_price=100.0,
                snapshot_ts=ref_ts,
            ))
    fit = infer_forward(quotes, ref_ts=ref_ts - 0.25 * 365 * 86400)
    assert fit is None


def test_surface_term_structure_and_rr_finite():
    snap = load_snapshot(FIXTURE)
    surf = build_surface(snap.for_underlying("BTC"), ref_ts=snap.collected_ts)
    # At least most slices produce a finite 25-delta RR/BF.
    n_rr = sum(1 for s in surf.slices if s.rr_25 is not None)
    assert n_rr >= len(surf.slices) - 2
    for s in surf.slices:
        if s.rr_25 is not None:
            assert abs(s.rr_25) < 0.5   # RR within 50 vol points (sane)
            assert s.bf_25 is not None


# --------------------------------------------------- market put-call parity (DoD #4)
def test_market_parity_residual_within_registered_bound():
    """Put-call parity holds on the REAL market snapshots within TOL['parity_market_resid_coin'].

    Design goal: "documented bounded parity residuals on market snapshots." Each expiry's
    ForwardFit.resid_rms is the RMS of C-P vs df*(F-K) over the used strike pairs, in USD;
    we convert to coin (÷ index) to compare against the coin-denominated reporting bound.
    This is the gate that certifies the parity-inferred forward actually fits the quotes.
    """
    bound = TOL["parity_market_resid_coin"].value
    checked = 0
    for path in _SNAPSHOTS:
        snap = load_snapshot(path)
        for ccy in ("BTC", "ETH"):
            index = snap.index_prices[ccy]
            fits = infer_forwards_all_expiries(snap.for_underlying(ccy), ref_ts=snap.collected_ts)
            for expiry_ts, fit in fits.items():
                resid_coin = fit.resid_rms / index
                assert resid_coin <= bound, (
                    f"{ccy} expiry {expiry_ts}: parity residual {resid_coin:.4f} coin "
                    f"exceeds bound {bound} (investigate stale/wide quotes, do not widen)"
                )
                checked += 1
    assert checked >= 20, f"expected many liquid expiries across snapshots, checked {checked}"


# --------------------------------------------------- SVI fit-RMSE reporting threshold
def test_svi_fit_rmse_liquid_slices_within_report_threshold():
    """Liquid real slices fit within TOL['svi_fit_rmse_report']; the known sparse/wide
    mid-tenor outlier (25Sep, stale far-wing marks — see DESIGN.md) is SURFACED not hidden.

    This wires the previously-unused reporting threshold: it is descriptive (not pass/fail
    for the whole surface), so we assert the liquid majority fit under it and require the
    outliers to be few and explicitly countable, matching how DESIGN.md describes the fit.
    """
    threshold = TOL["svi_fit_rmse_report"].value
    snap = load_snapshot(FIXTURE)
    for ccy in ("BTC", "ETH"):
        surf = build_surface(snap.for_underlying(ccy), ref_ts=snap.collected_ts)
        over = [s for s in surf.slices if s.svi.rmse_w > threshold]
        # The vast majority of liquid slices fit under the reporting threshold; only the
        # documented stale-far-wing-mark slice(s) may exceed it.
        assert len(over) <= 2, (
            f"{ccy}: {len(over)} slices over rmse_w {threshold} "
            f"(expected <=2 documented outliers): "
            + ", ".join(f"{s.expiry_ts}:{s.svi.rmse_w:.1e}" for s in over)
        )


# --------------------------------------------------- SVI w(k) >= 0 on real fitted slices
def test_svi_total_variance_nonnegative_on_real_slices():
    """The fitted SVI total variance stays >= 0 across the scanned wing grid for EVERY real
    calibrated slice — this is what constraint_wpos (a + b*sigma*sqrt(1-rho^2) >= 0) buys us.

    A regression that broke the constraint wiring would still fit and pass the synthetic
    single-param test; asserting it on the real fitted params over a wide k-grid guards it.
    """
    grid = np.linspace(-1.5, 1.5, 121)  # spans well past the liquid wings
    for path in _SNAPSHOTS:
        snap = load_snapshot(path)
        for ccy in ("BTC", "ETH"):
            surf = build_surface(snap.for_underlying(ccy), ref_ts=snap.collected_ts)
            for s in surf.slices:
                w = svi_total_variance(grid, s.svi.params)
                assert np.all(w >= -1e-12), (
                    f"{ccy} expiry {s.expiry_ts}: negative total variance "
                    f"min={float(np.min(w)):.3e} at fitted params {s.svi.params.as_dict()}"
                )


# --------------------------------------------------- degenerate: only-calls expiry
def test_only_calls_expiry_yields_no_forward():
    """An expiry with no put at any strike has zero C/P pairs, so parity inference must
    return None (honest-unknown) rather than fabricate a forward from calls alone."""
    ref_ts = 0.0
    quotes = []
    for k in (90.0, 100.0, 110.0, 120.0):  # calls only
        usd = BS.price(spot=100.0, strike=k, tau=0.25, rate=0.0, sigma=0.6,
                       option_type="C").price
        coin = usd / 100.0
        quotes.append(OptionQuote(
            instrument_name=f"BTC-X-{int(k)}-C", underlying="BTC", option_type="C",
            strike=k, expiry_ts=1_750_000_000.0, bid_coin=coin * 0.99, ask_coin=coin * 1.01,
            mark_price_coin=coin, mark_price_usd=usd, mark_iv=0.6, open_interest=10.0,
            index_price=100.0, underlying_price=100.0, snapshot_ts=ref_ts,
        ))
    fit = infer_forward(quotes, ref_ts=ref_ts - 0.25 * 365 * 86400)
    assert fit is None
