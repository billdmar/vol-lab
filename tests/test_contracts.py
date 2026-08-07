"""W0 contract smoke tests — exercise the frozen schema/interfaces/tolerances.

These lock the contract surface (provenance fields, derived properties, tolerance
registry integrity) before any engine is written, and give CI a green, fully-covered
baseline. Engine-specific tests arrive with their owning subagents in W1+.
"""

from __future__ import annotations

import math

import pytest

from config.tolerances import TOL, Tol, summary
from src.interfaces import Greeks, PriceResult
from src.schema import OptionQuote, Snapshot, SurfacePoint


def _quote(bid, ask, mark_coin=0.05, index=60000.0, oi=12.0):
    return OptionQuote(
        instrument_name="BTC-27JUN25-60000-C",
        underlying="BTC",
        option_type="C",
        strike=60000.0,
        expiry_ts=1_750_000_000.0,
        bid_coin=bid,
        ask_coin=ask,
        mark_price_coin=mark_coin,
        mark_price_usd=mark_coin * index,
        mark_iv=0.65,
        open_interest=oi,
        index_price=index,
        underlying_price=index * 1.001,
        snapshot_ts=1_749_000_000.0,
    )


def test_option_quote_derived_full_book():
    q = _quote(bid=0.048, ask=0.052)
    assert q.mid_coin == pytest.approx(0.05)
    assert q.spread_coin == pytest.approx(0.004)
    assert q.rel_spread == pytest.approx(0.004 / 0.05)
    assert q.mark_price_usd == pytest.approx(0.05 * 60000.0)


def test_option_quote_derived_missing_sides():
    # Missing bid -> mid/spread/rel_spread all None (honest-unknown, not fabricated).
    q = _quote(bid=None, ask=0.052)
    assert q.mid_coin is None
    assert q.spread_coin is None
    assert q.rel_spread is None
    # Missing ask too
    q2 = _quote(bid=0.048, ask=None)
    assert q2.mid_coin is None


def test_rel_spread_zero_mark_guard():
    q = _quote(bid=0.0, ask=0.0, mark_coin=0.0)
    assert q.rel_spread is None  # guarded division by zero mark


def test_snapshot_grouping_and_expiries():
    q_btc = _quote(bid=0.048, ask=0.052)
    q_eth = OptionQuote(
        instrument_name="ETH-27JUN25-3000-P",
        underlying="ETH",
        option_type="P",
        strike=3000.0,
        expiry_ts=1_760_000_000.0,
        bid_coin=0.01,
        ask_coin=0.012,
        mark_price_coin=0.011,
        mark_price_usd=0.011 * 3200.0,
        mark_iv=None,
        open_interest=3.0,
        index_price=3200.0,
        underlying_price=3200.0,
        snapshot_ts=1_749_000_000.0,
    )
    snap = Snapshot(
        collected_ts=1_749_000_000.0,
        index_prices={"BTC": 60000.0, "ETH": 3200.0},
        quotes=(q_btc, q_eth),
        deribit_server_ts_ms=1_749_000_000_000,
        meta={"collector": "vol-lab/0.1"},
    )
    assert snap.for_underlying("BTC") == (q_btc,)
    assert snap.for_underlying("ETH") == (q_eth,)
    assert snap.expiries("BTC") == (1_750_000_000.0,)
    assert snap.expiries("ETH") == (1_760_000_000.0,)


def test_surface_point_roundtrip_iv_variance():
    tau = 0.25
    iv = 0.6
    w = iv * iv * tau
    sp = SurfacePoint(
        underlying="BTC",
        expiry_ts=1_750_000_000.0,
        tau=tau,
        forward=60500.0,
        log_moneyness=math.log(60000.0 / 60500.0),
        strike=60000.0,
        total_variance=w,
        model_iv=math.sqrt(w / tau),
        market_iv=0.61,
        mark_iv=0.605,
        snapshot_ts=1_749_000_000.0,
    )
    assert sp.model_iv == pytest.approx(iv)
    assert sp.log_moneyness < 0.0  # strike below forward


def test_dataclasses_are_frozen():
    q = _quote(bid=0.048, ask=0.052)
    with pytest.raises((AttributeError, Exception)):
        q.strike = 1.0  # type: ignore[misc]


def test_priceresult_and_greeks_defaults():
    pr = PriceResult(price=1.23)
    assert pr.stderr is None and pr.ci95 is None
    pr2 = PriceResult(price=1.23, stderr=0.01, ci95=(1.21, 1.25))
    assert pr2.ci95 == (1.21, 1.25)
    g = Greeks(delta=0.5, gamma=0.01, vega=10.0, theta=-5.0, rho=2.0)
    assert g.delta == 0.5


# --------------------------------------------------------------- tolerance registry


def test_tolerance_registry_integrity():
    assert TOL, "registry must not be empty"
    valid_kinds = {"abs", "rel", "order", "prob"}
    for name, t in TOL.items():
        assert isinstance(t, Tol)
        assert t.kind in valid_kinds, f"{name} has bad kind {t.kind}"
        assert t.why.strip(), f"{name} missing justification"
        assert len(t.why) >= 40, f"{name} justification too thin to be real"
        if t.kind == "prob":
            assert 0.0 <= t.value <= 1.0


def test_tolerance_registry_has_required_gates():
    # The gates the mission explicitly names must exist by contract.
    required = {
        "parity_model",
        "crr_bs_convergence_price",
        "crr_convergence_order_min",
        "mc_ci_coverage_prob",
        "greeks_fd_vs_closed_rel",
        "greeks_mc_vs_closed_rel",
        "lsmc_vs_crr_american_rel",
        "svi_butterfly_g_min",
    }
    assert required <= set(TOL), f"missing gates: {required - set(TOL)}"


def test_tolerance_summary_renders():
    s = summary()
    assert "parity_model" in s
    assert s.count("\n") >= len(TOL)  # header + one line per entry
