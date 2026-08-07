"""Liquidity filters for Deribit option quotes (SA-data, owned path).

Illiquid quotes (no open interest, one-sided books, blown-out spreads) pollute an
IV surface. We drop them, but NEVER silently: every filter returns both the surviving
quotes AND a `FilterStats` object recording how many were dropped and why, so the
research note can report exactly what was excluded and readers can audit it.

Filter criteria (each with a written rationale):
  * min_open_interest : drop quotes with open_interest < threshold. Zero-OI strikes
    have no real market and their marks are stale/model-derived.
  * max_rel_spread    : drop quotes whose relative bid-ask spread (spread / mark, in
    coin units) exceeds the threshold. Wide spreads mean the mid is untrustworthy.
  * require_two_sided : drop quotes missing a bid or an ask (rel_spread is undefined),
    when a max_rel_spread bound is active.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.schema import OptionQuote


@dataclass(frozen=True, slots=True)
class FilterStats:
    """Accounting for a liquidity-filter pass: nothing is dropped without a reason.

    ``n_in`` quotes entered; ``n_out`` survived; ``by_reason`` maps each drop reason
    to a count. By construction ``n_in == n_out + sum(by_reason.values())``.
    """

    n_in: int
    n_out: int
    by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_in - self.n_out

    def check(self) -> None:
        """Assert the accounting balances (guards against a lost/double-counted quote)."""
        total_dropped = sum(self.by_reason.values())
        if self.n_out + total_dropped != self.n_in:
            raise AssertionError(
                f"FilterStats does not balance: in={self.n_in} out={self.n_out} "
                f"dropped={total_dropped}"
            )


def filter_quotes(
    quotes: tuple[OptionQuote, ...] | list[OptionQuote],
    *,
    min_open_interest: float = 0.0,
    max_rel_spread: float | None = None,
    require_two_sided: bool = False,
) -> tuple[tuple[OptionQuote, ...], FilterStats]:
    """Apply liquidity filters, returning (kept_quotes, FilterStats).

    A quote is checked against each active criterion in a fixed order; the FIRST
    failing criterion is recorded as its drop reason (so counts sum to n_dropped
    without double-counting). All thresholds default to no-op so callers opt in.

    ``min_open_interest`` keeps quotes with open_interest >= threshold.
    ``max_rel_spread`` keeps quotes with rel_spread <= threshold; a quote whose
    rel_spread is undefined (missing bid/ask or non-positive mark) is dropped as
    "no_rel_spread" only when this bound is active OR ``require_two_sided`` is set.
    """
    kept: list[OptionQuote] = []
    reasons: Counter[str] = Counter()

    for q in quotes:
        if q.open_interest < min_open_interest:
            reasons["low_open_interest"] += 1
            continue

        need_spread = max_rel_spread is not None or require_two_sided
        if need_spread:
            rs = q.rel_spread
            if rs is None:
                reasons["no_rel_spread"] += 1
                continue
            if max_rel_spread is not None and rs > max_rel_spread:
                reasons["wide_spread"] += 1
                continue

        kept.append(q)

    stats = FilterStats(n_in=len(quotes), n_out=len(kept), by_reason=dict(reasons))
    stats.check()
    return tuple(kept), stats
