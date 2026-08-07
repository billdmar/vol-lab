"""Tests for the Deribit data layer (SA-data owned: src/deribit/**).

Covers instrument-name parsing (golden hand-verified cases incl. ETH and edge
strikes, malformed rejection), snapshot loading against the REAL committed fixture
(inverse-contract USD conversion, percent->decimal IV, null book -> None), liquidity
filters with drop accounting, and per-expiry grouping.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os

import pytest

from src.deribit import (
    FilterStats,
    distinct_expiries,
    filter_quotes,
    group_by_expiry,
    load_snapshot,
    parse_instrument_name,
)
from src.deribit.parse import _parse_expiry, _parse_strike
from src.deribit.store import _mark_iv_decimal, _to_opt_float
from src.schema import OptionQuote, Snapshot

FIXTURE_DIR = "data/snapshots"


def _fixture_path() -> str:
    matches = sorted(glob.glob(os.path.join(FIXTURE_DIR, "snapshot_*.json")))
    assert matches, "no committed snapshot fixture found under data/snapshots/"
    return matches[0]


# --------------------------------------------------------------- instrument parsing


def test_golden_parse_btc_put():
    # Golden #1: hand-verified. 28 Aug 2026 08:00 UTC == 1787904000.
    p = parse_instrument_name("BTC-28AUG26-110000-P")
    assert p.underlying == "BTC"
    assert p.option_type == "P"
    assert p.strike == 110000.0
    assert p.expiry == dt.datetime(2026, 8, 28, 8, 0, 0, tzinfo=dt.UTC)
    assert p.expiry_ts == 1787904000.0


def test_golden_parse_eth_call_single_digit_day():
    # Golden #2: ETH, single-digit day. 8 Aug 2026 08:00 UTC == 1786176000.
    p = parse_instrument_name("ETH-8AUG26-1800-C")
    assert p.underlying == "ETH"
    assert p.option_type == "C"
    assert p.strike == 1800.0
    assert p.expiry_ts == 1786176000.0


def test_golden_parse_far_dated_and_high_strike():
    # Golden #3: far-dated 2027 expiry, high strike. 25 Jun 2027 08:00 UTC == 1813910400.
    p = parse_instrument_name("BTC-25JUN27-320000-C")
    assert p.strike == 320000.0
    assert p.expiry_ts == 1813910400.0
    # And a December expiry as a fourth hand-checked case.
    p2 = parse_instrument_name("BTC-25DEC26-64000-P")
    assert p2.expiry == dt.datetime(2026, 12, 25, 8, 0, 0, tzinfo=dt.UTC)
    assert p2.expiry_ts == 1798185600.0


def test_parse_expiry_hour_is_0800_utc():
    # Deribit settlement instant is 08:00 UTC, not midnight — guards a silent regression.
    p = parse_instrument_name("BTC-10AUG26-60000-C")
    assert p.expiry.hour == 8
    assert p.expiry.tzinfo == dt.UTC


def test_parse_lowercase_type_char_normalized():
    p = parse_instrument_name("BTC-28AUG26-110000-p")
    assert p.option_type == "P"


@pytest.mark.parametrize(
    "name",
    [
        "BTC-28AUG26-110000",          # only 3 fields (future/perp shape)
        "BTC-28AUG26-110000-P-EXTRA",  # 5 fields
        "SOL-28AUG26-110000-P",        # unsupported underlying
        "BTC-28XYZ26-110000-P",        # bad month code
        "BTC-28AUG26-abc-P",           # non-numeric strike
        "BTC-28AUG26-110000-X",        # bad type char
        "BTC-32AUG26-110000-P",        # invalid calendar day
        "BTC--110000-P",               # empty expiry token
        "BTC-28AUG26--P",              # empty strike
        "BTC-28AUG26-0-P",             # non-positive strike
        "BTC-28AUG26--100-P",          # negative strike (empty field split artifact)
    ],
)
def test_malformed_names_raise(name):
    with pytest.raises(ValueError):
        parse_instrument_name(name)


def test_non_string_name_raises():
    with pytest.raises(ValueError):
        parse_instrument_name(12345)  # type: ignore[arg-type]


def test_negative_strike_rejected():
    with pytest.raises(ValueError):
        _parse_strike("-100")


def test_parse_expiry_too_short_raises():
    with pytest.raises(ValueError):
        _parse_expiry("AUG26")  # no day digits


def test_parse_expiry_nondigit_year_raises():
    with pytest.raises(ValueError):
        _parse_expiry("1AUGZZ")  # month ok, year not numeric


# --------------------------------------------------------------- unit conversions


def test_percent_to_decimal_iv():
    assert _mark_iv_decimal(65.75) == pytest.approx(0.6575)
    assert _mark_iv_decimal(100.0) == pytest.approx(1.0)


def test_zero_or_missing_iv_is_none():
    # 0 and null are Deribit's "no mark IV" — honest-unknown, not a fabricated 0% vol.
    assert _mark_iv_decimal(0.0) is None
    assert _mark_iv_decimal(None) is None
    assert _mark_iv_decimal(-1.0) is None


def test_null_bid_ask_maps_to_none():
    assert _to_opt_float(None) is None
    assert _to_opt_float(0.687) == pytest.approx(0.687)
    assert _to_opt_float(0) == 0.0


# --------------------------------------------------------------- real-fixture load


def test_load_real_fixture_counts_match_file():
    path = _fixture_path()
    with open(path) as f:
        raw = json.load(f)
    snap = load_snapshot(path)
    assert isinstance(snap, Snapshot)

    # File's own "counts" field is ground truth for per-underlying totals.
    counts = raw["counts"]
    total_expected = sum(counts.values())
    assert total_expected == 1540, "fixture should hold the documented 1540-instrument board"
    assert len(snap.quotes) == total_expected
    assert len(snap.for_underlying("BTC")) == counts["BTC"]
    assert len(snap.for_underlying("ETH")) == counts["ETH"]


def test_load_real_fixture_inverse_conversion_and_iv():
    path = _fixture_path()
    with open(path) as f:
        raw = json.load(f)
    snap = load_snapshot(path)

    # Pick the first BTC row and reconstruct the conversion by hand.
    first_raw = raw["book_summaries"]["BTC"][0]
    idx = raw["index_prices"]["BTC"]
    q = next(x for x in snap.quotes if x.instrument_name == first_raw["instrument_name"])

    # Inverse contract: USD premium = coin mark * index price.
    assert q.mark_price_coin == pytest.approx(first_raw["mark_price"])
    assert q.mark_price_usd == pytest.approx(first_raw["mark_price"] * idx)
    assert q.index_price == pytest.approx(idx)

    # mark_iv percent -> decimal.
    if first_raw.get("mark_iv"):
        assert q.mark_iv == pytest.approx(first_raw["mark_iv"] / 100.0)

    # underlying_price passed through as USD.
    assert q.underlying_price == pytest.approx(first_raw["underlying_price"])


def test_load_real_fixture_null_bid_becomes_none():
    path = _fixture_path()
    with open(path) as f:
        raw = json.load(f)
    snap = load_snapshot(path)

    # Find a raw row with a null bid and assert the parsed quote has bid_coin None.
    null_bid = next(
        r for u in raw["book_summaries"].values() for r in u if r.get("bid_price") is None
    )
    q = next(x for x in snap.quotes if x.instrument_name == null_bid["instrument_name"])
    assert q.bid_coin is None
    assert q.mid_coin is None  # derived property honors the missing side


def test_load_real_fixture_all_quotes_wellformed():
    snap = load_snapshot(_fixture_path())
    for q in snap.quotes:
        assert q.underlying in ("BTC", "ETH")
        assert q.option_type in ("C", "P")
        assert q.strike > 0.0
        assert q.expiry_ts > 0.0
        assert q.mark_price_coin >= 0.0
        assert q.snapshot_ts > 0.0
        if q.mark_iv is not None:
            assert q.mark_iv > 0.0


def test_snapshot_provenance_populated():
    snap = load_snapshot(_fixture_path())
    assert snap.collected_ts > 0.0
    assert set(snap.index_prices) == {"BTC", "ETH"}
    assert snap.deribit_server_ts_ms is not None
    assert snap.meta.get("collector")


def test_per_row_snapshot_ts_from_creation_timestamp():
    path = _fixture_path()
    with open(path) as f:
        raw = json.load(f)
    snap = load_snapshot(path)
    row = raw["book_summaries"]["BTC"][0]
    q = next(x for x in snap.quotes if x.instrument_name == row["instrument_name"])
    if row.get("creation_timestamp"):
        assert q.snapshot_ts == pytest.approx(row["creation_timestamp"] / 1000.0)


# --------------------------------------------------------------- liquidity filters


def _mk_quote(name, oi, bid, ask, mark_coin, index=60000.0):
    from src.deribit.parse import parse_instrument_name as _p

    p = _p(name)
    return OptionQuote(
        instrument_name=name,
        underlying=p.underlying,
        option_type=p.option_type,
        strike=p.strike,
        expiry_ts=p.expiry_ts,
        bid_coin=bid,
        ask_coin=ask,
        mark_price_coin=mark_coin,
        mark_price_usd=mark_coin * index,
        mark_iv=0.65,
        open_interest=oi,
        index_price=index,
        underlying_price=index,
        snapshot_ts=1_749_000_000.0,
    )


def test_filter_noop_by_default():
    quotes = (
        _mk_quote("BTC-28AUG26-60000-C", oi=0.0, bid=None, ask=0.01, mark_coin=0.02),
        _mk_quote("BTC-28AUG26-61000-C", oi=5.0, bid=0.01, ask=0.02, mark_coin=0.015),
    )
    kept, stats = filter_quotes(quotes)
    assert len(kept) == 2
    assert stats.n_dropped == 0
    assert stats.by_reason == {}


def test_filter_min_open_interest():
    quotes = (
        _mk_quote("BTC-28AUG26-60000-C", oi=0.0, bid=0.01, ask=0.02, mark_coin=0.015),
        _mk_quote("BTC-28AUG26-61000-C", oi=10.0, bid=0.01, ask=0.02, mark_coin=0.015),
    )
    kept, stats = filter_quotes(quotes, min_open_interest=1.0)
    assert len(kept) == 1
    assert kept[0].open_interest == 10.0
    assert stats.by_reason == {"low_open_interest": 1}
    stats.check()


def test_filter_wide_spread_and_missing_side():
    quotes = (
        # tight spread: 0.001/0.02 = 0.05 rel
        _mk_quote("BTC-28AUG26-60000-C", oi=5.0, bid=0.0195, ask=0.0205, mark_coin=0.02),
        # wide spread: 0.02/0.02 = 1.0 rel
        _mk_quote("BTC-28AUG26-61000-C", oi=5.0, bid=0.01, ask=0.03, mark_coin=0.02),
        # missing bid -> rel_spread undefined
        _mk_quote("BTC-28AUG26-62000-C", oi=5.0, bid=None, ask=0.03, mark_coin=0.02),
    )
    kept, stats = filter_quotes(quotes, max_rel_spread=0.2)
    assert len(kept) == 1
    assert kept[0].instrument_name == "BTC-28AUG26-60000-C"
    assert stats.by_reason == {"wide_spread": 1, "no_rel_spread": 1}
    stats.check()


def test_filter_require_two_sided_without_spread_bound():
    quotes = (
        _mk_quote("BTC-28AUG26-60000-C", oi=5.0, bid=0.01, ask=0.02, mark_coin=0.015),
        _mk_quote("BTC-28AUG26-61000-C", oi=5.0, bid=None, ask=0.02, mark_coin=0.015),
    )
    kept, stats = filter_quotes(quotes, require_two_sided=True)
    assert len(kept) == 1
    assert stats.by_reason == {"no_rel_spread": 1}


def test_filter_reason_order_first_failure_wins():
    # Quote fails both low OI and wide spread; only the first checked (OI) is counted.
    q = _mk_quote("BTC-28AUG26-60000-C", oi=0.0, bid=0.01, ask=0.09, mark_coin=0.02)
    kept, stats = filter_quotes((q,), min_open_interest=1.0, max_rel_spread=0.1)
    assert kept == ()
    assert stats.by_reason == {"low_open_interest": 1}
    assert sum(stats.by_reason.values()) == stats.n_dropped


def test_filter_stats_balance_on_real_fixture():
    snap = load_snapshot(_fixture_path())
    kept, stats = filter_quotes(snap.quotes, min_open_interest=1.0, max_rel_spread=0.5)
    assert stats.n_in == len(snap.quotes)
    assert stats.n_out == len(kept)
    stats.check()  # in == out + dropped
    # Filtering the real board must actually remove some illiquid strikes.
    assert stats.n_dropped > 0


def test_filter_stats_check_detects_imbalance():
    bad = FilterStats(n_in=10, n_out=5, by_reason={"low_open_interest": 2})
    with pytest.raises(AssertionError):
        bad.check()


# --------------------------------------------------------------- grouping


def test_distinct_expiries_sorted_unique():
    snap = load_snapshot(_fixture_path())
    btc = snap.for_underlying("BTC")
    exps = distinct_expiries(btc)
    assert list(exps) == sorted(set(exps))
    assert exps == snap.expiries("BTC")


def test_group_by_expiry_deterministic_order():
    quotes = (
        _mk_quote("BTC-28AUG26-61000-P", oi=5.0, bid=0.01, ask=0.02, mark_coin=0.015),
        _mk_quote("BTC-28AUG26-60000-C", oi=5.0, bid=0.01, ask=0.02, mark_coin=0.015),
        _mk_quote("BTC-8AUG26-60000-C", oi=5.0, bid=0.01, ask=0.02, mark_coin=0.015),
    )
    grouped = group_by_expiry(quotes)
    keys = list(grouped)
    assert keys == sorted(keys)  # ascending expiry
    aug28 = grouped[parse_instrument_name("BTC-28AUG26-60000-C").expiry_ts]
    # Within a bucket: sorted by (strike, option_type) -> 60000-C before 61000-P.
    assert [q.instrument_name for q in aug28] == [
        "BTC-28AUG26-60000-C",
        "BTC-28AUG26-61000-P",
    ]


def test_group_by_expiry_covers_all_quotes_on_fixture():
    snap = load_snapshot(_fixture_path())
    grouped = group_by_expiry(snap.quotes)
    assert sum(len(v) for v in grouped.values()) == len(snap.quotes)
    assert set(grouped) == set(distinct_expiries(snap.quotes))


def test_group_by_expiry_empty():
    assert group_by_expiry(()) == {}
    assert distinct_expiries(()) == ()
