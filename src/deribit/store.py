"""Load raw Deribit snapshot fixtures into typed `Snapshot`/`OptionQuote` objects.

This is the ONLY interpreter of the raw capture written by
``scripts/collect_snapshot.py``. The raw file is untouched market data; every
modeling convention applied here is documented so a published statistic can be
traced back to the exact quote and conversion that produced it.

Conventions applied (rationale inline):
  * mark_iv is stored by the exchange in PERCENT (e.g. 65.75); we divide by 100 to
    the project-wide decimal convention. A missing or zero mark_iv becomes ``None``
    (honest-unknown: 0 is Deribit's "no vol available", not a real 0% vol).
  * bid_price / ask_price of ``null`` (empty book side) become ``None``.
  * INVERSE-CONTRACT conversion: Deribit option premiums are quoted in units of the
    underlying COIN (BTC/ETH). The USD premium is coin premium x index price. We keep
    BOTH the coin mark and the converted USD mark so the conversion is auditable and
    never silently applied (schema retains `mark_price_coin` and `mark_price_usd`).
  * expiry_ts comes from the parsed instrument name (08:00 UTC settlement); the
    per-quote snapshot_ts uses the row's `creation_timestamp` (ms) when present,
    else the snapshot's `collected_ts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.deribit.parse import parse_instrument_name
from src.schema import OptionQuote, Snapshot


def _to_opt_float(value: Any) -> float | None:
    """Coerce a JSON number to float, mapping null -> None (empty book side)."""
    if value is None:
        return None
    return float(value)


def _mark_iv_decimal(raw: Any) -> float | None:
    """Convert Deribit's percent mark_iv to a decimal, or None if absent/zero.

    Deribit reports mark_iv in percent (65.75 == 65.75% == 0.6575). A ``null`` or
    ``0`` is the exchange saying "no mark IV available" for that quote, which we
    surface as None rather than a fabricated 0.0 vol.
    """
    if raw is None:
        return None
    iv = float(raw)
    if iv <= 0.0:
        return None
    return iv / 100.0


def _build_quote(row: dict[str, Any], index_price: float, collected_ts: float) -> OptionQuote:
    """Build one `OptionQuote` from a raw book_summary row for a known underlying."""
    name = row["instrument_name"]
    parsed = parse_instrument_name(name)

    mark_price_coin = float(row["mark_price"])
    # Inverse-contract conversion: coin premium x index price -> USD premium.
    mark_price_usd = mark_price_coin * index_price

    # Per-row creation_timestamp is in milliseconds; fall back to the snapshot start.
    creation_ms = row.get("creation_timestamp")
    snapshot_ts = float(creation_ms) / 1000.0 if creation_ms else collected_ts

    return OptionQuote(
        instrument_name=name,
        underlying=parsed.underlying,
        option_type=parsed.option_type,
        strike=parsed.strike,
        expiry_ts=parsed.expiry_ts,
        bid_coin=_to_opt_float(row.get("bid_price")),
        ask_coin=_to_opt_float(row.get("ask_price")),
        mark_price_coin=mark_price_coin,
        mark_price_usd=mark_price_usd,
        mark_iv=_mark_iv_decimal(row.get("mark_iv")),
        open_interest=float(row.get("open_interest", 0.0)),
        index_price=index_price,
        underlying_price=float(row["underlying_price"]),
        snapshot_ts=snapshot_ts,
    )


def load_snapshot(path: str | Path) -> Snapshot:
    """Parse a raw snapshot JSON fixture into an immutable `Snapshot`.

    Builds an `OptionQuote` for every book_summary row across all underlyings in the
    file, applying the percent->decimal IV, null->None book, and inverse-contract USD
    conversions documented in this module. Raises on a malformed instrument name (the
    parser refuses to guess) rather than dropping it silently.
    """
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    index_prices: dict[str, float] = {k: float(v) for k, v in raw["index_prices"].items()}
    collected_ts = float(raw["collected_ts"])

    quotes: list[OptionQuote] = []
    for underlying, rows in raw["book_summaries"].items():
        # index_prices is keyed by underlying ("BTC"/"ETH"); each row also carries its
        # own underlying via the instrument name, which the parser validates.
        idx = index_prices[underlying]
        for row in rows:
            quotes.append(_build_quote(row, idx, collected_ts))

    server_ts = raw.get("deribit_server_ts_ms")
    meta: dict[str, str] = {}
    if "collector" in raw:
        meta["collector"] = str(raw["collector"])
    if "schema_version" in raw:
        meta["schema_version"] = str(raw["schema_version"])
    if "collected_iso" in raw:
        meta["collected_iso"] = str(raw["collected_iso"])

    return Snapshot(
        collected_ts=collected_ts,
        index_prices=index_prices,
        quotes=tuple(quotes),
        deribit_server_ts_ms=int(server_ts) if server_ts is not None else None,
        meta=meta,
    )
