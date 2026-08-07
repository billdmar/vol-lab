"""Vol-lab figure toolkit — house style + deterministic showcase plots.

The public entry point is ``scripts/make_figures.py``; this package holds the
consistent house style (:mod:`src.viz.style`) shared by every figure so the
showcase reads as one visual system.
"""

from src.viz.style import (
    CATEGORICAL,
    PALETTE,
    SEQ_BLUE,
    SEQ_ORANGE,
    SERIES,
    apply_house_style,
    footer,
    sample_ramp,
    savefig,
)

__all__ = [
    "CATEGORICAL",
    "PALETTE",
    "SEQ_BLUE",
    "SEQ_ORANGE",
    "SERIES",
    "apply_house_style",
    "footer",
    "sample_ramp",
    "savefig",
]
