"""Static no-arbitrage scanner for calibrated SVI surfaces and raw market quotes (W2/G2).

Three model-free (or SVI-parametric) checks, each QUANTIFIED and never smoothed:

  1. Butterfly / convexity arbitrage
       * SVI slice: Gatheral's Durrleman function g(k). For raw SVI total variance
             w(k) = a + b(rho*(k-m) + sqrt((k-m)^2 + sigma^2)),
         a butterfly-arb-free slice needs g(k) >= 0 everywhere, where
             g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2.
         The derivatives are analytic (see `svi_w_derivatives`). We scan a dense k-grid
         and flag any g(k) < TOL["svi_butterfly_g_min"].value, reporting the min g and its k.
       * Raw quotes: undiscounted call prices must be convex in strike. For adjacent
         strikes K1<K2<K3 the right-minus-left slope
             (C3-C2)/(K3-K2) - (C2-C1)/(K2-K1)
         must be >= 0 (a long butterfly has non-negative value). Violations are reported
         in price-units-per-strike with the three strikes involved.

  2. Calendar arbitrage
       Total implied variance must be non-decreasing in tau at fixed log-moneyness k:
       w(k, tau_inner) <= w(k, tau_outer) for tau_inner < tau_outer. We sample a k-grid on
       the overlap of each adjacent expiry pair's calibration range and flag any crossing,
       reporting the total-variance gap w_inner - w_outer and the (k, tau-pair) location.

Everything is deterministic (fixed grids, no RNG). The scanner is descriptive on real
markets: it RUNS and QUANTIFIES violations with plausible causes; it does not assert zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.tolerances import TOL
from src.surface.smiles import Smile
from src.surface.surface import Surface
from src.surface.svi import SVIParams, svi_total_variance

_G_MIN = TOL["svi_butterfly_g_min"].value


# --------------------------------------------------------------------------- SVI derivatives


def svi_w_derivatives(k, params: SVIParams):
    """Analytic (w, w', w'') of raw-SVI total variance at log-moneyness k.

    With u = k - m and s = sqrt(u^2 + sigma^2):
        w   = a + b*(rho*u + s)
        w'  = b*(rho + u/s)
        w'' = b*sigma^2 / s^3
    Accepts scalar or array k; returns three arrays.
    """
    k = np.asarray(k, dtype=float)
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    u = k - m
    s = np.sqrt(u * u + sigma * sigma)
    w = a + b * (rho * u + s)
    wp = b * (rho + u / s)
    wpp = b * sigma * sigma / (s * s * s)
    return w, wp, wpp


def durrleman_g(k, params: SVIParams):
    """Gatheral's Durrleman g(k) for a raw-SVI slice (analytic derivatives).

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2.
    g(k) >= 0 everywhere is the butterfly-arbitrage-free condition. Returns an array.
    """
    w, wp, wpp = svi_w_derivatives(k, params)
    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = (wp * wp / 4.0) * (1.0 / w + 0.25)
    return term1 - term2 + wpp / 2.0


# --------------------------------------------------------------------------- result records


@dataclass(frozen=True, slots=True)
class SliceButterflyResult:
    """Durrleman g(k) scan on one calibrated SVI slice."""

    expiry_ts: float
    tau: float
    g_min: float          # minimum of g(k) over the scanned grid
    k_at_min: float       # log-moneyness where the minimum occurs
    n_grid: int
    violated: bool        # g_min < TOL["svi_butterfly_g_min"].value


@dataclass(frozen=True, slots=True)
class PriceButterflyViolation:
    """One call-price convexity violation among adjacent strikes (raw quotes)."""

    expiry_ts: float
    tau: float
    strikes: tuple[float, float, float]   # (K1, K2, K3)
    convexity: float                      # right-slope minus left-slope (price/strike)
    magnitude: float                      # -convexity > 0 (violation size, price/strike)
    price_units: str                      # "USD" (undiscounted call premium)


@dataclass(frozen=True, slots=True)
class CalendarViolation:
    """One total-variance crossing between two adjacent expiries at fixed k."""

    k: float
    tau_inner: float
    tau_outer: float
    expiry_inner: float
    expiry_outer: float
    w_inner: float
    w_outer: float
    gap: float            # w_inner - w_outer > 0 (violation magnitude, total-variance units)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Aggregated no-arbitrage scan for one underlying's surface (+ optional raw quotes)."""

    underlying: str
    # --- butterfly on the calibrated SVI slices (Durrleman g) ---
    slice_butterfly: tuple[SliceButterflyResult, ...]
    n_slice_butterfly_violations: int
    worst_g_min: float | None             # most-negative g across slices (None if no slices)
    worst_g_location: tuple[float, float] | None   # (tau, k) of the worst g
    # --- butterfly on raw call prices (convexity in strike) ---
    price_butterfly: tuple[PriceButterflyViolation, ...]
    n_price_butterfly_violations: int
    worst_price_butterfly: PriceButterflyViolation | None
    # --- calendar across adjacent expiries ---
    calendar: tuple[CalendarViolation, ...]
    n_calendar_violations: int
    worst_calendar_gap: float | None      # max positive gap (None if no pairs checked)
    worst_calendar_location: tuple[float, float, float] | None  # (k, tau_inner, tau_outer)

    @property
    def total_violations(self) -> int:
        return (
            self.n_slice_butterfly_violations
            + self.n_price_butterfly_violations
            + self.n_calendar_violations
        )


# --------------------------------------------------------------------------- SVI butterfly


def scan_slice_butterfly(
    params: SVIParams,
    tau: float,
    expiry_ts: float,
    *,
    k_lo: float = -1.5,
    k_hi: float = 1.5,
    n_grid: int = 601,
) -> SliceButterflyResult:
    """Scan Durrleman g(k) on one SVI slice over a dense k-grid.

    The default grid spans +/-1.5 in log-moneyness (well past the liquid wings) so a
    convexity failure anywhere in the traded region is caught. Reports the minimum g and
    its k; `violated` if that minimum drops below the (float-noise) tolerance band.
    """
    grid = np.linspace(k_lo, k_hi, n_grid)
    g = durrleman_g(grid, params)
    i = int(np.argmin(g))
    g_min = float(g[i])
    return SliceButterflyResult(
        expiry_ts=expiry_ts,
        tau=tau,
        g_min=g_min,
        k_at_min=float(grid[i]),
        n_grid=n_grid,
        violated=g_min < _G_MIN,
    )


# --------------------------------------------------------------------------- price butterfly


def scan_price_butterfly(
    smile: Smile,
    *,
    call_prices: dict[float, float] | None = None,
) -> list[PriceButterflyViolation]:
    """Check convexity of undiscounted call prices in strike for one expiry.

    `call_prices` maps strike -> USD call premium; if omitted, it is reconstructed from the
    smile's forward + IVs via Black-Scholes so the check works off the calibrated smile.
    We use the smile's own points (each is a solved market IV) to rebuild an arb-consistent
    call curve when raw quote prices are not supplied. Returns one record per adjacent
    triplet whose right-slope < left-slope (a negative-value butterfly).
    """
    prices = call_prices if call_prices is not None else _call_prices_from_smile(smile)
    strikes = sorted(prices)
    out: list[PriceButterflyViolation] = []
    for i in range(1, len(strikes) - 1):
        k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
        c1, c2, c3 = prices[k1], prices[k2], prices[k3]
        left = (c2 - c1) / (k2 - k1)
        right = (c3 - c2) / (k3 - k2)
        convexity = right - left
        if convexity < 0.0:
            out.append(PriceButterflyViolation(
                expiry_ts=smile.expiry_ts,
                tau=smile.tau,
                strikes=(k1, k2, k3),
                convexity=float(convexity),
                magnitude=float(-convexity),
                price_units="USD",
            ))
    return out


def _call_prices_from_smile(smile: Smile) -> dict[float, float]:
    """Rebuild undiscounted USD call prices at each smile strike from its solved IVs.

    Uses forward-measure Black-Scholes with the slice forward F and rate r, so the curve
    is exactly the one implied by the market IVs at those strikes (put IVs are converted to
    the same-strike call via the arb-free BS map at that IV). Convexity of THIS curve is a
    clean read on whether the fitted smile is butterfly-consistent strike-by-strike.
    """
    from src.bs import BS

    spot = smile.forward * np.exp(-smile.rate * smile.tau)
    prices: dict[float, float] = {}
    for p in smile.points:
        res = BS.price(
            spot=spot, strike=p.strike, tau=smile.tau, rate=smile.rate,
            sigma=p.iv, option_type="C", carry=0.0,
        )
        prices[p.strike] = float(res.price)
    return prices


# --------------------------------------------------------------------------- calendar


def scan_calendar(
    surface: Surface,
    *,
    n_grid: int = 41,
    pad: float = 0.0,
) -> list[CalendarViolation]:
    """Check total variance is non-decreasing in tau at fixed k across adjacent expiries.

    Slices are ordered by tau; for each adjacent pair we sample k on the OVERLAP of their
    calibration ranges (so we compare where both smiles have data, not extrapolation), and
    flag any k where w_inner(k) > w_outer(k). `pad` shrinks the overlap symmetrically if a
    conservative in-sample window is wanted. Reports the total-variance gap per crossing.
    """
    slices = sorted(surface.slices, key=lambda s: s.tau)
    out: list[CalendarViolation] = []
    for inner, outer in zip(slices[:-1], slices[1:], strict=False):
        ki = inner.smile.log_moneyness
        ko = outer.smile.log_moneyness
        if len(ki) == 0 or len(ko) == 0:
            continue
        lo = max(float(ki.min()), float(ko.min())) + pad
        hi = min(float(ki.max()), float(ko.max())) - pad
        if not (hi > lo):
            continue
        grid = np.linspace(lo, hi, n_grid)
        w_in = svi_total_variance(grid, inner.svi.params)
        w_out = svi_total_variance(grid, outer.svi.params)
        gap = w_in - w_out
        for j in range(len(grid)):
            if gap[j] > 0.0:
                out.append(CalendarViolation(
                    k=float(grid[j]),
                    tau_inner=inner.tau,
                    tau_outer=outer.tau,
                    expiry_inner=inner.expiry_ts,
                    expiry_outer=outer.expiry_ts,
                    w_inner=float(w_in[j]),
                    w_outer=float(w_out[j]),
                    gap=float(gap[j]),
                ))
    return out


# --------------------------------------------------------------------------- top-level scan


def scan_surface(
    surface: Surface,
    *,
    scan_prices: bool = True,
) -> ScanResult:
    """Run all no-arb checks on one calibrated surface and aggregate into a ScanResult.

    Butterfly is checked on every SVI slice (Durrleman g) and, when `scan_prices`, on the
    call-price convexity of each slice's smile. Calendar is checked across adjacent expiries.
    Counts, worst magnitudes+locations, and per-slice/per-pair detail are all retained.
    """
    slice_bf = tuple(
        scan_slice_butterfly(s.svi.params, s.tau, s.expiry_ts) for s in surface.slices
    )
    n_slice_bf = sum(1 for r in slice_bf if r.violated)
    worst_g_min: float | None = None
    worst_g_loc: tuple[float, float] | None = None
    if slice_bf:
        worst = min(slice_bf, key=lambda r: r.g_min)
        worst_g_min = worst.g_min
        worst_g_loc = (worst.tau, worst.k_at_min)

    price_bf: list[PriceButterflyViolation] = []
    if scan_prices:
        for s in surface.slices:
            price_bf.extend(scan_price_butterfly(s.smile))
    worst_price = max(price_bf, key=lambda v: v.magnitude) if price_bf else None

    cal = scan_calendar(surface)
    worst_cal_gap: float | None = None
    worst_cal_loc: tuple[float, float, float] | None = None
    if cal:
        w = max(cal, key=lambda v: v.gap)
        worst_cal_gap = w.gap
        worst_cal_loc = (w.k, w.tau_inner, w.tau_outer)

    return ScanResult(
        underlying=surface.underlying,
        slice_butterfly=slice_bf,
        n_slice_butterfly_violations=n_slice_bf,
        worst_g_min=worst_g_min,
        worst_g_location=worst_g_loc,
        price_butterfly=tuple(price_bf),
        n_price_butterfly_violations=len(price_bf),
        worst_price_butterfly=worst_price,
        calendar=tuple(cal),
        n_calendar_violations=len(cal),
        worst_calendar_gap=worst_cal_gap,
        worst_calendar_location=worst_cal_loc,
    )
