"""Implied-volatility surface construction: parity forwards, smiles, SVI calibration.

Pipeline (one underlying, one snapshot): infer the forward per expiry from put-call
parity (`forwards`), invert market mids to a smile in log-moneyness/delta space
(`smiles`), calibrate a raw-SVI slice under no-arbitrage constraints (`svi`), and
assemble the term structure + 25-delta risk-reversal/butterfly (`surface`).
"""

from src.surface.forwards import (
    ForwardFit,
    infer_forward,
    infer_forwards_all_expiries,
    year_fraction,
)
from src.surface.smiles import Smile, SmilePoint, build_smile
from src.surface.surface import ExpirySlice, Surface, build_surface
from src.surface.svi import (
    SVIFit,
    SVIParams,
    calibrate_svi,
    svi_iv,
    svi_total_variance,
)

__all__ = [
    "ExpirySlice",
    "ForwardFit",
    "SVIFit",
    "SVIParams",
    "Smile",
    "SmilePoint",
    "Surface",
    "build_smile",
    "build_surface",
    "calibrate_svi",
    "infer_forward",
    "infer_forwards_all_expiries",
    "svi_iv",
    "svi_total_variance",
    "year_fraction",
]
