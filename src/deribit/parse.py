"""Deribit instrument-name parsing (SA-data, owned path).

Deribit option instrument names follow a fixed four-field, dash-delimited grammar:

    <UNDERLYING>-<EXPIRY>-<STRIKE>-<TYPE>

e.g. "BTC-28AUG26-110000-P"  ->  BTC, 28 Aug 2026, strike 110000, put.

Field grammar (verified against a live board fixture):
  * UNDERLYING : "BTC" or "ETH" (the two underlyings this project covers).
  * EXPIRY     : day(1-2 digits) + 3-letter month + 2-digit year, e.g. "8AUG26",
                 "28AUG26". Deribit options ALWAYS expire at 08:00:00 UTC on that
                 date (the daily settlement instant), which we encode explicitly so
                 downstream tau computations are exact rather than assuming midnight.
  * STRIKE     : USD strike, integer on the live board (parsed as float for the
                 schema, which stores strike as float).
  * TYPE       : "C" (call) or "P" (put).

Anything that does not match this grammar raises `ValueError` — we never silently
coerce a malformed name into a plausible-but-wrong instrument (honest-unknown rule).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src.schema import OptionType

# Deribit options settle at 08:00 UTC on the expiry date (exchange convention).
_EXPIRY_HOUR_UTC = 8

_MONTHS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_UNDERLYINGS = ("BTC", "ETH")


@dataclass(frozen=True, slots=True)
class ParsedInstrument:
    """The four structured fields decoded from a Deribit option instrument name."""

    underlying: str          # "BTC" or "ETH"
    expiry: dt.datetime      # tz-aware UTC datetime at 08:00 UTC on the expiry date
    strike: float            # USD strike
    option_type: OptionType  # "C" or "P"

    @property
    def expiry_ts(self) -> float:
        """POSIX seconds (UTC) of the 08:00-UTC expiry instant."""
        return self.expiry.timestamp()


def _parse_expiry(token: str) -> dt.datetime:
    """Decode an expiry token like '28AUG26' -> 2026-08-28 08:00:00 UTC.

    Day is 1-2 digits, month is a 3-letter uppercase code, year is 2 digits
    (interpreted as 20YY — Deribit has no pre-2000 or post-2099 options).
    """
    if len(token) < 6:  # min: 1 day digit + 3 month + 2 year
        raise ValueError(f"expiry token too short: {token!r}")

    month_code = token[-5:-2]
    year_str = token[-2:]
    day_str = token[:-5]

    if month_code not in _MONTHS:
        raise ValueError(f"unknown month code {month_code!r} in expiry {token!r}")
    if not (day_str.isdigit() and year_str.isdigit()):
        raise ValueError(f"malformed day/year in expiry token {token!r}")

    day = int(day_str)
    year = 2000 + int(year_str)
    month = _MONTHS[month_code]
    try:
        return dt.datetime(year, month, day, _EXPIRY_HOUR_UTC, 0, 0, tzinfo=dt.UTC)
    except ValueError as exc:  # e.g. day 31 in a 30-day month
        raise ValueError(f"invalid calendar date in expiry token {token!r}: {exc}") from exc


def _parse_strike(token: str) -> float:
    """Decode a strike token; must be a positive number."""
    try:
        strike = float(token)
    except ValueError as exc:
        raise ValueError(f"non-numeric strike {token!r}") from exc
    if strike <= 0.0:
        raise ValueError(f"non-positive strike {token!r}")
    return strike


def parse_instrument_name(name: str) -> ParsedInstrument:
    """Parse a Deribit option instrument name into its structured fields.

    Raises ``ValueError`` on any name that does not match the four-field option
    grammar (wrong field count, unknown underlying, bad date, non-numeric strike,
    or a type char other than C/P) — futures/perps/combos are rejected, not guessed.
    """
    if not isinstance(name, str):
        raise ValueError(f"instrument name must be a string, got {type(name).__name__}")

    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"expected 4 dash-separated fields, got {len(parts)} in {name!r} "
            "(not an option instrument?)"
        )

    underlying, expiry_tok, strike_tok, type_tok = parts

    if underlying not in _UNDERLYINGS:
        raise ValueError(f"unsupported underlying {underlying!r} in {name!r}")

    option_type = type_tok.upper()
    if option_type not in ("C", "P"):
        raise ValueError(f"option type must be C or P, got {type_tok!r} in {name!r}")

    expiry = _parse_expiry(expiry_tok)
    strike = _parse_strike(strike_tok)

    return ParsedInstrument(
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]  # validated above
    )
