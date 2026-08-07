"""Frozen data contracts for vol-lab (W0 — ORCH-owned, do not edit in subagents).

These dataclasses are the shared vocabulary between the Deribit data layer, the
pricing engines, and the surface/calibration code. Every record carries enough
*provenance* (instrument name, snapshot timestamp, index price) that any published
statistic can be traced back to the exact quote it came from.

Conventions frozen here (rationale in docs/DESIGN.md):
  * Option type is the single char "C" or "P".
  * Times are POSIX seconds (UTC). Year fractions use ACT/365 (see interfaces/tolerances).
  * Deribit option premiums are quoted in units of the underlying *coin* (inverse
    contracts). `mark_price_coin` is the coin-denominated mark; `mark_price_usd` is
    the converted USD premium (coin premium x index price). Both are retained so the
    conversion is auditable and never silently applied.
  * Implied vols are decimals (0.65 == 65%), never percent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

OptionType = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One option instrument's market data at a single snapshot instant.

    Prices in `*_coin` fields are in coin units (Deribit inverse convention); the
    `*_usd` fields are the index-converted USD premiums. `mark_iv` is Deribit's own
    published mark implied vol (decimal) — the external ground truth for our solver.
    """

    instrument_name: str          # e.g. "BTC-27JUN25-60000-C"
    underlying: str               # "BTC" or "ETH"
    option_type: OptionType       # "C" or "P"
    strike: float                 # USD strike
    expiry_ts: float              # POSIX seconds (UTC) of expiry
    # --- market quote (coin-denominated, Deribit native) ---
    bid_coin: float | None        # best bid in coin, None if no bid
    ask_coin: float | None        # best ask in coin, None if no ask
    mark_price_coin: float        # Deribit mark price in coin
    # --- converted / exchange-published fields ---
    mark_price_usd: float         # mark_price_coin * index_price
    mark_iv: float | None         # Deribit published mark IV (decimal), None if absent
    # --- liquidity / provenance ---
    open_interest: float          # contracts
    index_price: float            # Deribit index (USD) for the underlying at snapshot
    underlying_price: float       # Deribit per-instrument underlying (forward-ish), USD
    snapshot_ts: float            # POSIX seconds (UTC) when this snapshot was taken

    @property
    def mid_coin(self) -> float | None:
        """Coin-denominated mid, or None if either side of the book is empty."""
        if self.bid_coin is None or self.ask_coin is None:
            return None
        return 0.5 * (self.bid_coin + self.ask_coin)

    @property
    def spread_coin(self) -> float | None:
        """Absolute bid-ask spread in coin, or None if one side is missing."""
        if self.bid_coin is None or self.ask_coin is None:
            return None
        return self.ask_coin - self.bid_coin

    @property
    def rel_spread(self) -> float | None:
        """Bid-ask spread relative to mark (coin units cancel), or None."""
        s = self.spread_coin
        if s is None or self.mark_price_coin <= 0.0:
            return None
        return s / self.mark_price_coin


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A full point-in-time capture of the Deribit options board for the underlyings.

    `quotes` holds every instrument row. `index_prices` maps underlying -> index USD
    at capture time. `collected_ts` is when collection *started* (per-quote
    `snapshot_ts` may differ by the polite request spacing).
    """

    collected_ts: float                       # POSIX seconds (UTC), collection start
    index_prices: dict[str, float]            # {"BTC": 60123.4, "ETH": 3210.9}
    quotes: tuple[OptionQuote, ...]           # immutable; all instruments in this snap
    deribit_server_ts_ms: int | None = None   # exchange /public/get_time at capture
    meta: dict[str, str] = field(default_factory=dict)  # collector version, UA, etc.

    def for_underlying(self, underlying: str) -> tuple[OptionQuote, ...]:
        return tuple(q for q in self.quotes if q.underlying == underlying)

    def expiries(self, underlying: str) -> tuple[float, ...]:
        return tuple(sorted({q.expiry_ts for q in self.quotes if q.underlying == underlying}))


@dataclass(frozen=True, slots=True)
class SurfacePoint:
    """One calibrated point on the implied-volatility surface.

    Carries the model IV from our SVI fit alongside the inputs that produced it, so
    the exchange differential (our_iv vs `mark_iv`) is always reconstructable.
    """

    underlying: str
    expiry_ts: float
    tau: float                 # year fraction to expiry (ACT/365)
    forward: float             # parity-inferred forward (USD)
    log_moneyness: float       # ln(strike / forward)
    strike: float              # USD
    total_variance: float      # w = sigma^2 * tau  (SVI's native quantity)
    model_iv: float            # our calibrated IV (decimal) = sqrt(w / tau)
    market_iv: float | None    # IV implied from the market mid by our solver
    mark_iv: float | None      # Deribit published mark IV (decimal)
    snapshot_ts: float         # provenance back to the source snapshot
