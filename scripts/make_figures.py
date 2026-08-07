#!/usr/bin/env python3
"""Deterministic showcase-figure generator for vol-lab (SA-viz, G3).

Regenerates every showcase PNG under docs/figures/ from the committed snapshot
fixtures and the calibrated surface / verification engines. Seeded and headless
(Agg backend) so the figures reproduce identically.

Figures:
    01_surface_3d_{ccy}       3D SVI implied-vol surface (log-moneyness x tau x IV)
    02_smile_overlay_{ccy}    market IV points + fitted SVI curve for a few expiries
    03_atm_term_structure     ATM vol vs tau, BTC and ETH on one axes
    04_rr_bf_25delta          25-delta risk-reversal and butterfly vs tau
    05_crr_convergence        |CRR - BS| vs N log-log with measured order
    06_exchange_differential  histogram of our_iv - mark_iv, BTC and ETH

Usage:
    python scripts/make_figures.py                 # latest snapshot -> docs/figures/
    python scripts/make_figures.py --outdir /tmp/f # write elsewhere
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np

# Run-as-script: put the repo root on sys.path so `src` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless, before pyplot

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3d projection)

from src.deribit.store import load_snapshot  # noqa: E402
from src.exchdiff.differential import run_exchange_differential  # noqa: E402
from src.lattice import convergence_order  # noqa: E402
from src.surface.surface import build_surface  # noqa: E402
from src.surface.svi import svi_iv  # noqa: E402
from src.viz.style import (  # noqa: E402
    PALETTE,
    SEQ_BLUE,
    SERIES,
    apply_house_style,
    footer,
    sample_ramp,
    savefig,
)

CCYS = ("BTC", "ETH")


def _expiry_label(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%d%b%y")


def _date_str(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Figure 1 — 3D implied-vol surface from the calibrated SVI slices.
# --------------------------------------------------------------------------- #
def fig_surface_3d(surf, ccy: str, date_str: str, outdir: str) -> str:
    apply_house_style()
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # A common log-moneyness grid, clipped to the union of observed smile ranges
    # so the SVI surface is drawn only where the calibration was informed by data.
    k_lo = max(min(p.log_moneyness for p in s.smile.points) for s in surf.slices)
    k_hi = min(max(p.log_moneyness for p in s.smile.points) for s in surf.slices)
    k_grid = np.linspace(k_lo, k_hi, 60)

    taus = np.array([s.tau for s in surf.slices])
    K, T = np.meshgrid(k_grid, taus)
    Z = np.empty_like(K)
    for i, s in enumerate(surf.slices):
        Z[i, :] = [svi_iv(float(k), s.svi.params, s.tau) * 100.0 for k in k_grid]

    surfc = ax.plot_surface(
        K,
        T,
        Z,
        cmap="viridis",
        linewidth=0.2,
        edgecolor=PALETTE["surface"],
        antialiased=True,
        alpha=0.95,
        rcount=len(taus),
        ccount=40,
    )
    ax.set_xlabel("log-moneyness  ln(K/F)", labelpad=10)
    ax.set_ylabel("time to expiry  τ (yr)", labelpad=10)
    ax.set_zlabel("implied vol (%)", labelpad=8)
    ax.set_title(f"{ccy} implied-volatility surface — calibrated SVI")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)
    # Recede the 3D panes to the house surface tone.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.988, 0.988, 0.984, 1.0))
    cb = fig.colorbar(surfc, ax=ax, shrink=0.55, pad=0.10)
    cb.set_label("implied vol (%)", color=PALETTE["ink_secondary"])
    cb.ax.tick_params(colors=PALETTE["muted"])

    footer(fig, date_str)
    path = os.path.join(outdir, f"01_surface_3d_{ccy}.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 2 — smile overlays: market IV points + fitted SVI curve.
# --------------------------------------------------------------------------- #
def fig_smile_overlay(surf, ccy: str, date_str: str, outdir: str) -> str:
    apply_house_style()
    # Pick up to four expiries spread across the term structure (short..long).
    slices = surf.slices
    n_pick = min(4, len(slices))
    idx = [round(i * (len(slices) - 1) / (n_pick - 1)) for i in range(n_pick)]
    picked = [slices[j] for j in idx]
    # Ordinal ramp: start no lighter than step 300 (palette ordinal-contrast rule)
    # so the shortest-expiry curve stays legible on the light surface.
    colors = sample_ramp(SEQ_BLUE[2:], n_pick)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s, c in zip(picked, colors, strict=True):
        ks = np.array([p.log_moneyness for p in s.smile.points])
        ivs = np.array([p.iv for p in s.smile.points]) * 100.0
        order = np.argsort(ks)
        ks, ivs = ks[order], ivs[order]
        label = f"{_expiry_label(s.expiry_ts)}  (τ={s.tau:.3f})"
        ax.scatter(ks, ivs, s=22, color=c, alpha=0.75, edgecolor=PALETTE["surface"],
                   linewidth=0.4, zorder=3)
        kf = np.linspace(ks.min(), ks.max(), 200)
        fit = np.array([svi_iv(float(k), s.svi.params, s.tau) for k in kf]) * 100.0
        ax.plot(kf, fit, color=c, linewidth=2.0, zorder=2, label=label)

    ax.axvline(0.0, color=PALETTE["axis"], linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("log-moneyness  ln(K/F)")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"{ccy} volatility smiles — market points vs fitted SVI")
    ax.legend(title="expiry", loc="upper center", ncol=2)
    fig.text(0.13, 0.02, "dots = solved market IV   line = SVI fit",
             fontsize=8, color=PALETTE["muted"])
    footer(fig, date_str)
    path = os.path.join(outdir, f"02_smile_overlay_{ccy}.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 3 — ATM term structure, BTC and ETH on one axes.
# --------------------------------------------------------------------------- #
def fig_atm_term_structure(surfaces: dict, date_str: str, outdir: str) -> str:
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for ccy in CCYS:
        surf = surfaces[ccy]
        ts = np.array([tau for tau, _ in surf.term_structure])
        vols = np.array([v for _, v in surf.term_structure]) * 100.0
        ax.plot(ts, vols, color=SERIES[ccy], marker="o", markersize=5,
                markeredgecolor=PALETTE["surface"], markeredgewidth=0.5, label=ccy)
    ax.set_xlabel("time to expiry  τ (yr)")
    ax.set_ylabel("ATM implied vol (%)")
    ax.set_title("ATM term structure — BTC and ETH")
    ax.legend(loc="best")
    footer(fig, date_str)
    path = os.path.join(outdir, "03_atm_term_structure.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 4 — 25-delta risk-reversal and butterfly vs tau.
# --------------------------------------------------------------------------- #
def fig_rr_bf(surfaces: dict, date_str: str, outdir: str) -> str:
    apply_house_style()
    fig, (ax_rr, ax_bf) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ccy in CCYS:
        surf = surfaces[ccy]
        rr_pts = [(s.tau, s.rr_25 * 100.0) for s in surf.slices if s.rr_25 is not None]
        bf_pts = [(s.tau, s.bf_25 * 100.0) for s in surf.slices if s.bf_25 is not None]
        if rr_pts:
            t, y = zip(*sorted(rr_pts), strict=True)
            ax_rr.plot(t, y, color=SERIES[ccy], marker="o", markersize=5,
                       markeredgecolor=PALETTE["surface"], markeredgewidth=0.5, label=ccy)
        if bf_pts:
            t, y = zip(*sorted(bf_pts), strict=True)
            ax_bf.plot(t, y, color=SERIES[ccy], marker="o", markersize=5,
                       markeredgecolor=PALETTE["surface"], markeredgewidth=0.5, label=ccy)

    ax_rr.axhline(0.0, color=PALETTE["axis"], linewidth=0.8, linestyle="--")
    ax_rr.set_xlabel("time to expiry  τ (yr)")
    ax_rr.set_ylabel("25Δ risk-reversal (vol pts)")
    ax_rr.set_title("25-delta risk-reversal")
    ax_rr.legend(loc="best")

    ax_bf.set_xlabel("time to expiry  τ (yr)")
    ax_bf.set_ylabel("25Δ butterfly (vol pts)")
    ax_bf.set_title("25-delta butterfly")
    ax_bf.legend(loc="best")

    fig.suptitle("25-delta skew and convexity vs expiry — BTC and ETH",
                 fontsize=12, fontweight="bold", color=PALETTE["ink"])
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.24)
    footer(fig, date_str)
    path = os.path.join(outdir, "04_rr_bf_25delta.png")
    savefig(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 5 — CRR -> BS convergence on log-log with measured order.
# --------------------------------------------------------------------------- #
def fig_crr_convergence(outdir: str, date_str: str) -> tuple[str, float]:
    apply_house_style()
    conv = convergence_order(
        spot=100.0, strike=100.0, tau=1.0, rate=0.02, sigma=0.2, option_type="C"
    )
    ns = np.array(conv["ns"], dtype=float)
    errs = np.array(conv["errors"], dtype=float)
    order = float(conv["order"])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.loglog(ns, errs, color=SERIES["BTC"], marker="o", markersize=6,
              markeredgecolor=PALETTE["surface"], markeredgewidth=0.5,
              label="|CRR(N) − BS|")
    # Reference O(1/N) guide anchored to the first point.
    guide = errs[0] * ns[0] / ns
    ax.loglog(ns, guide, color=PALETTE["muted"], linewidth=1.4, linestyle="--",
              label="O(1/N) reference")

    ax.set_xlabel("tree steps  N")
    ax.set_ylabel("absolute pricing error (USD)")
    ax.set_title("CRR binomial → Black-Scholes convergence")
    ax.grid(True, which="both", linewidth=0.6)
    ax.annotate(
        f"measured order ≈ {order:.4f}\n"
        "European call, S=K=100, τ=1, r=2%, σ=20%",
        xy=(0.04, 0.06), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=9, color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#eef4fc",
              "edgecolor": SERIES["BTC"], "linewidth": 0.8},
    )
    ax.legend(loc="upper right")
    footer(fig, date_str)
    path = os.path.join(outdir, "05_crr_convergence.png")
    savefig(fig, path)
    return path, order


# --------------------------------------------------------------------------- #
# Figure 6 — exchange differential histogram (our_iv - mark_iv).
# --------------------------------------------------------------------------- #
def fig_exchange_differential(snap, date_str: str, outdir: str) -> str:
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=False)
    for ax, ccy in zip(axes, CCYS, strict=True):
        exch = run_exchange_differential(snap, ccy)
        deltas = np.array([p.delta_sigma for p in exch.points]) * 100.0  # vol pts
        o = exch.overall
        med, q25, q75 = o.median * 100.0, o.q25 * 100.0, o.q75 * 100.0

        # Symmetric range around zero, clipped to the bulk so wings don't flatten it.
        lim = float(np.percentile(np.abs(deltas), 99)) if deltas.size else 1.0
        lim = max(lim, 0.1)
        bins = np.linspace(-lim, lim, 41)
        ax.hist(deltas, bins=bins, color=SERIES[ccy], alpha=0.85,
                edgecolor=PALETTE["surface"], linewidth=0.4)
        ax.axvline(0.0, color=PALETTE["axis"], linewidth=0.8, linestyle="--")
        ax.axvline(med, color=PALETTE["ink"], linewidth=1.6,
                   label=f"median {med:+.3f}")
        ax.axvspan(q25, q75, color=PALETTE["ink"], alpha=0.08,
                   label=f"IQR {o.iqr * 100:.3f}")
        ax.set_xlabel("our IV − Deribit mark IV (vol pts)")
        ax.set_ylabel("count")
        ax.set_title(f"{ccy}  (n={exch.n_matched})")
        ax.legend(loc="upper right")

    fig.suptitle("Exchange differential — our solved IV vs Deribit mark IV",
                 fontsize=12, fontweight="bold", color=PALETTE["ink"])
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.22)
    footer(fig, date_str)
    path = os.path.join(outdir, "06_exchange_differential.png")
    savefig(fig, path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate vol-lab showcase figures.")
    ap.add_argument("--outdir", default=None, help="output dir (default docs/figures)")
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir or os.path.join(repo_root, "docs", "figures")
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(repo_root, "data", "snapshots", "snapshot_*.json")))
    if not paths:
        print("no snapshots found under data/snapshots/", file=sys.stderr)
        return 1
    snap = load_snapshot(paths[-1])  # latest snapshot
    date_str = _date_str(snap.collected_ts)

    surfaces = {ccy: build_surface(snap.for_underlying(ccy), ref_ts=snap.collected_ts)
                for ccy in CCYS}

    written: list[str] = []
    for ccy in CCYS:
        written.append(fig_surface_3d(surfaces[ccy], ccy, date_str, outdir))
        written.append(fig_smile_overlay(surfaces[ccy], ccy, date_str, outdir))
    written.append(fig_atm_term_structure(surfaces, date_str, outdir))
    written.append(fig_rr_bf(surfaces, date_str, outdir))
    conv_path, order = fig_crr_convergence(outdir, date_str)
    written.append(conv_path)
    written.append(fig_exchange_differential(snap, date_str, outdir))

    print(f"snapshot: {os.path.basename(paths[-1])}  ({date_str})")
    print(f"measured CRR->BS convergence order: {order:.4f}")
    print(f"wrote {len(written)} figures to {outdir}:")
    for p in written:
        size = os.path.getsize(p)
        print(f"  {os.path.basename(p):32s} {size:>8,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
