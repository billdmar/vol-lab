#!/usr/bin/env python3
"""Deribit snapshot collector (ORCH-owned, W0) — raw capture only.

Captures one point-in-time snapshot of the BTC + ETH options board from Deribit's
PUBLIC API (no auth) and writes it as raw JSON under data/snapshots/. SA-data's
`src/deribit` layer parses these raw fixtures into typed `Snapshot`/`OptionQuote`
objects — this script does NOT parse or interpret, so there is exactly one writer
per concern (capture here, structured parsing in src/deribit).

Deribit etiquette (CLAUDE.md hard rules), all enforced below:
  * public endpoints only, no auth
  * descriptive User-Agent
  * >= 250ms spacing between requests
  * every response cached to disk (committed as fixtures); CI never calls live

Efficiency: `get_book_summary_by_currency(kind=option)` returns the ENTIRE board for a
currency in one response (bid/ask/mark/mark_iv/open_interest/underlying_price), so a
full snapshot is ~5 requests total rather than one-per-instrument — the politest way
to get complete coverage.

Resumable: writes a timestamped file per run and skips collection if a snapshot for
the same UTC day already exists (override with --force). Safe to relaunch on later
build days to accumulate the >= 5 distinct snapshot days the mission targets.

Usage:
    python scripts/collect_snapshot.py                # collect if none today
    python scripts/collect_snapshot.py --force        # always collect
    python scripts/collect_snapshot.py --dir data/snapshots
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.deribit.com/api/v2/public"
USER_AGENT = "vol-lab/0.1 (research; github.com/billdmar/vol-lab)"
MIN_SPACING_S = 0.25  # >= 250ms between requests (Deribit etiquette)
CURRENCIES = ("BTC", "ETH")
MAX_RETRIES = 4       # attempts per request before giving up
BACKOFF_BASE_S = 0.5  # exponential backoff: 0.5s, 1s, 2s, ... between retries
RETRY_STATUS = {429, 500, 502, 503, 504}  # transient HTTP codes worth retrying

_last_request_ts = 0.0


def _throttle() -> None:
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < MIN_SPACING_S:
        time.sleep(MIN_SPACING_S - elapsed)
    _last_request_ts = time.time()


def _get(method: str, **params) -> dict:
    """Throttled GET against a public Deribit endpoint with retry/backoff on transient errors.

    Retries on timeouts, connection errors (`URLError`), and transient HTTP status codes
    (429 rate-limit, 5xx) with exponential backoff, still honoring the >=250ms spacing on
    every attempt. A non-transient HTTP error (e.g. 400) or an exhausted retry budget
    raises — the collector fails loudly rather than writing a partial/fabricated snapshot.
    """
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{method}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https host)
                payload = json.loads(resp.read().decode("utf-8"))
            if "result" not in payload:
                raise RuntimeError(f"{method}: unexpected payload {payload!r}")
            return payload["result"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in RETRY_STATUS:
                raise  # 4xx (except 429) is a real client error — don't retry.
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e  # transient network / decode blip — retry.
        if attempt < MAX_RETRIES - 1:
            wait = BACKOFF_BASE_S * (2**attempt)
            print(f"[collect] {method} attempt {attempt + 1}/{MAX_RETRIES} failed "
                  f"({last_err}); retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"{method}: failed after {MAX_RETRIES} attempts; last error: {last_err}")


def collect() -> dict:
    """Collect one full snapshot; returns the raw capture dict (also the file contents)."""
    collected_ts = time.time()
    server_time_ms = _get("get_time")
    index_prices = {}
    for ccy in CURRENCIES:
        r = _get("get_index_price", index_name=f"{ccy.lower()}_usd")
        index_prices[ccy] = r["index_price"]
    book_summaries = {}
    for ccy in CURRENCIES:
        book_summaries[ccy] = _get("get_book_summary_by_currency", currency=ccy, kind="option")

    return {
        "schema_version": 1,
        "collector": USER_AGENT,
        "collected_ts": collected_ts,
        "collected_iso": dt.datetime.fromtimestamp(collected_ts, dt.UTC).isoformat(),
        "deribit_server_ts_ms": server_time_ms,
        "index_prices": index_prices,
        # mark_iv fields in these summaries are PERCENT (e.g. 65.75); src/deribit /100s them.
        "book_summaries": book_summaries,
        "counts": {ccy: len(book_summaries[ccy]) for ccy in CURRENCIES},
    }


def _utc_day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y%m%d")


def _has_snapshot_today(snap_dir: str) -> bool:
    today = _utc_day(time.time())
    return bool(glob.glob(os.path.join(snap_dir, f"snapshot_{today}_*.json")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect one Deribit BTC+ETH options snapshot.")
    ap.add_argument("--dir", default="data/snapshots", help="output directory")
    ap.add_argument("--force", action="store_true", help="collect even if today already has one")
    args = ap.parse_args(argv)

    os.makedirs(args.dir, exist_ok=True)
    if not args.force and _has_snapshot_today(args.dir):
        print(f"[collect] snapshot for {_utc_day(time.time())} already exists; skipping "
              f"(use --force to override).")
        return 0

    print(f"[collect] fetching BTC+ETH options board from Deribit ({USER_AGENT}) ...")
    snap = collect()
    stamp = dt.datetime.fromtimestamp(snap["collected_ts"], dt.UTC).strftime("%Y%m%d_%H%M%SZ")
    out = os.path.join(args.dir, f"snapshot_{stamp}.json")
    # Atomic write: serialize to a temp file in the same dir, then os.replace() — a crash
    # mid-write can never leave a truncated fixture at the real path.
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f, separators=(",", ":"))
    os.replace(tmp, out)
    size_kb = os.path.getsize(out) / 1024.0
    print(f"[collect] wrote {out} ({size_kb:.0f} KB): "
          + ", ".join(f"{c}={snap['counts'][c]}" for c in CURRENCIES)
          + f"; index {snap['index_prices']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
