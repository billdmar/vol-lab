"""Exchange differential: OUR solver's implied vol vs Deribit's published mark IV.

For every option quote that carries a valid mark IV and a market mid, we already solve
an implied vol inside the surface build (each `SmilePoint.iv` is our BS-inverted vol on
the parity-consistent forward/rate for that expiry). This module matches each of those
solved vols to its source quote's `mark_iv` by (expiry, strike, option_type) and reports
the DISTRIBUTION of the differences

    delta_sigma = our_iv - mark_iv     (vol points, decimal)

overall and bucketed by moneyness (|log-moneyness|: ATM / near / wing) and expiry
(short < 30d / medium < 90d / long). Per the mission, this is descriptive-only — we never
claim agreement without printing the distribution (TOL["exch_diff_report_only"]).

Modeling notes:
  * We reuse the smile points' solved IV rather than re-inverting, so "our_iv" is exactly
    the vol that fed SVI calibration — the differential describes the same surface we ship.
  * The smile keeps the liquid OTM side per strike, so each (expiry, strike, type) key is
    unique and the match is 1:1; there is no ITM double-count.
  * Outlier causes are diagnosed, not asserted: wide spread, deep-OTM vega collapse,
    mark-fallback (no two-sided quote / possibly stale), else mark-IV construction diffs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schema import OptionQuote, Snapshot
from src.surface import build_surface

# Moneyness buckets on |ln(K/F)| (dimensionless). ATM = |ln(K/F)| < 0.05, i.e. strike
# within ~5% of the forward; near-money out to ~20%; the rest is the illiquid wing
# where vega collapses.
_ATM_ABS_K = 0.05
_NEAR_ABS_K = 0.20

# Expiry buckets in ACT/365 year fraction: 30d and 90d in years.
_SHORT_TAU = 30.0 / 365.0
_MEDIUM_TAU = 90.0 / 365.0

# Diagnosis thresholds (descriptive labels, not pass/fail gates).
_WIDE_REL_SPREAD = 0.5   # bid-ask > 50% of mark: illiquid, mid is unreliable
_WING_ABS_K = _NEAR_ABS_K  # |k| beyond the near band: OTM wing / vega collapse


@dataclass(frozen=True, slots=True)
class DiffPoint:
    """One matched (our_iv vs mark_iv) comparison with its provenance."""

    instrument_name: str
    underlying: str
    option_type: str
    strike: float
    expiry_ts: float
    tau: float
    log_moneyness: float
    our_iv: float
    mark_iv: float
    delta_sigma: float          # our_iv - mark_iv (vol points, decimal)
    rel_spread: float | None    # bid-ask / mark (coin), None if one-sided
    used_mid: bool              # True if a real two-sided mid produced our_iv


@dataclass(frozen=True, slots=True)
class DiffStats:
    """Distribution summary of delta_sigma over a set of matched points (vol points)."""

    n: int
    median: float               # median signed delta_sigma
    q25: float                  # 25th percentile of delta_sigma
    q75: float                  # 75th percentile of delta_sigma
    iqr: float                  # q75 - q25
    mean: float
    std: float                  # population std (ddof=0)
    median_abs: float           # median |delta_sigma| — the headline agreement number


@dataclass(frozen=True, slots=True)
class Outlier:
    """A large-|delta_sigma| point with a plausible, diagnosed cause."""

    point: DiffPoint
    abs_delta_sigma: float
    cause: str


@dataclass(frozen=True, slots=True)
class ExchDiffResult:
    """Full exchange-differential report for one underlying in one snapshot."""

    underlying: str
    snapshot_ts: float
    n_matched: int
    overall: DiffStats
    by_moneyness: dict[str, DiffStats]   # keys: "ATM", "near", "wing"
    by_expiry: dict[str, DiffStats]      # keys: "short", "medium", "long"
    outliers: tuple[Outlier, ...]
    points: tuple[DiffPoint, ...]


def _moneyness_bucket(abs_k: float) -> str:
    if abs_k < _ATM_ABS_K:
        return "ATM"
    if abs_k < _NEAR_ABS_K:
        return "near"
    return "wing"


def _expiry_bucket(tau: float) -> str:
    if tau < _SHORT_TAU:
        return "short"
    if tau < _MEDIUM_TAU:
        return "medium"
    return "long"


def _stats(deltas: np.ndarray) -> DiffStats:
    """Distribution summary of a delta_sigma array. Empty array -> all-NaN, n=0."""
    n = int(deltas.size)
    if n == 0:
        nan = float("nan")
        return DiffStats(n=0, median=nan, q25=nan, q75=nan, iqr=nan,
                         mean=nan, std=nan, median_abs=nan)
    q25, med, q75 = (float(x) for x in np.percentile(deltas, [25, 50, 75]))
    return DiffStats(
        n=n,
        median=med,
        q25=q25,
        q75=q75,
        iqr=q75 - q25,
        mean=float(np.mean(deltas)),
        std=float(np.std(deltas)),
        median_abs=float(np.median(np.abs(deltas))),
    )


def _diagnose(p: DiffPoint) -> str:
    """Plausible cause for a large |delta_sigma|, in priority order.

    These are diagnostic labels for the write-up, not assertions of fact — we lead with
    the most concrete observable (spread, mark-fallback) before the softer explanations.
    """
    if not p.used_mid:
        return ("mark-fallback: no two-sided mid, our IV was solved from Deribit's mark "
                "price itself, so the residual reflects float/rounding not a real diff "
                "(or a possibly stale mark)")
    if p.rel_spread is not None and p.rel_spread > _WIDE_REL_SPREAD:
        return (f"wide spread: rel_spread={p.rel_spread:.2f} (>{_WIDE_REL_SPREAD:.0%} of "
                "mark) — the mid is noisy, so our mid-IV drifts from the mark")
    if abs(p.log_moneyness) >= _WING_ABS_K:
        return (f"deep-OTM wing: |ln(K/F)|={abs(p.log_moneyness):.2f} — low vega means a "
                "tiny premium difference maps to a large vol difference (vega collapse)")
    return ("mark-IV construction difference: near-money with a tight book, so the "
            "residual likely reflects Deribit's mark-IV smoothing/timing vs our mid-IV")


def run_exchange_differential(
    snapshot: Snapshot,
    underlying: str,
    *,
    min_smile_points: int = 5,
    n_outliers: int = 15,
) -> ExchDiffResult:
    """Compute the our-IV vs mark-IV differential for one underlying in one snapshot.

    Builds the surface for `underlying` (which solves our IV per strike/type on the
    parity-consistent forward), matches each smile point to its quote's `mark_iv`, and
    returns the full distribution of delta_sigma = our_iv - mark_iv plus a diagnosed
    outlier list. Deterministic: sorting and numpy percentiles are order-stable.
    """
    quotes = snapshot.for_underlying(underlying)
    if not quotes:
        raise ValueError(f"no quotes for underlying {underlying!r}")

    # Match key -> quote. The OTM-only smile keeps one option per (expiry, strike, type),
    # so this lookup is unambiguous for the points we compare.
    by_key: dict[tuple[float, float, str], OptionQuote] = {
        (q.expiry_ts, q.strike, q.option_type): q for q in quotes
    }

    surface = build_surface(quotes, ref_ts=snapshot.collected_ts,
                            min_smile_points=min_smile_points)

    points: list[DiffPoint] = []
    for sl in surface.slices:
        for sp in sl.smile.points:
            q = by_key.get((sl.expiry_ts, sp.strike, sp.option_type))
            if q is None or q.mark_iv is None:
                continue
            delta = sp.iv - q.mark_iv
            points.append(DiffPoint(
                instrument_name=q.instrument_name,
                underlying=underlying,
                option_type=sp.option_type,
                strike=sp.strike,
                expiry_ts=sl.expiry_ts,
                tau=sl.tau,
                log_moneyness=sp.log_moneyness,
                our_iv=sp.iv,
                mark_iv=q.mark_iv,
                delta_sigma=delta,
                rel_spread=q.rel_spread,
                used_mid=sp.used_mid,
            ))

    # Stable order for determinism (expiry, strike, type) before slicing/statistics.
    points.sort(key=lambda p: (p.expiry_ts, p.strike, p.option_type))
    points_t = tuple(points)

    deltas = np.array([p.delta_sigma for p in points_t], dtype=float)
    overall = _stats(deltas)

    by_moneyness = {
        b: _stats(np.array([p.delta_sigma for p in points_t
                            if _moneyness_bucket(abs(p.log_moneyness)) == b], dtype=float))
        for b in ("ATM", "near", "wing")
    }
    by_expiry = {
        b: _stats(np.array([p.delta_sigma for p in points_t
                            if _expiry_bucket(p.tau) == b], dtype=float))
        for b in ("short", "medium", "long")
    }

    # Largest |delta_sigma| first; ties broken by the stable key for determinism.
    ranked = sorted(points_t, key=lambda p: (-abs(p.delta_sigma), p.expiry_ts,
                                             p.strike, p.option_type))
    outliers = tuple(
        Outlier(point=p, abs_delta_sigma=abs(p.delta_sigma), cause=_diagnose(p))
        for p in ranked[:n_outliers]
    )

    return ExchDiffResult(
        underlying=underlying,
        snapshot_ts=surface.snapshot_ts,
        n_matched=len(points_t),
        overall=overall,
        by_moneyness=by_moneyness,
        by_expiry=by_expiry,
        outliers=outliers,
        points=points_t,
    )
