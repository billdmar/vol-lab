"""Minimal grouping helpers over parsed quotes (SA-data, owned path).

The frozen `Snapshot` already exposes `for_underlying` and `expiries`; these add the
one thing W2's per-expiry smile calibration needs — grouping a quote list by expiry
into deterministically ordered buckets — without duplicating what the schema offers.
"""

from __future__ import annotations

from src.schema import OptionQuote


def distinct_expiries(quotes: tuple[OptionQuote, ...] | list[OptionQuote]) -> tuple[float, ...]:
    """Sorted, de-duplicated expiry timestamps across the given quotes."""
    return tuple(sorted({q.expiry_ts for q in quotes}))


def group_by_expiry(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
) -> dict[float, tuple[OptionQuote, ...]]:
    """Group quotes by expiry_ts.

    Returns an insertion-ordered dict keyed by ascending expiry; within each bucket
    quotes are ordered by (strike, option_type) so the output is fully deterministic
    regardless of input order — smile calibration wants a stable strike ladder.
    """
    buckets: dict[float, list[OptionQuote]] = {}
    for q in quotes:
        buckets.setdefault(q.expiry_ts, []).append(q)
    ordered: dict[float, tuple[OptionQuote, ...]] = {}
    for ts in sorted(buckets):
        rows = sorted(buckets[ts], key=lambda q: (q.strike, q.option_type))
        ordered[ts] = tuple(rows)
    return ordered
