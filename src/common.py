"""Shared numerical helpers used by more than one engine.

Small, dependency-light utilities used across the pricing engines: `intrinsic` by the
lattice and LSMC engines, `mean_stderr` by the Monte-Carlo and LSMC engines, and the
`DEFAULT_SEED` by both stochastic engines. Kept in one place so the definitions can't
drift apart (they were previously duplicated per engine while the ownership map was in
force during the build).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from src.schema import OptionType

# Global default RNG seed. Every stochastic engine accepts a `seed` argument and defaults
# to this so the whole project is reproducible from one documented value.
DEFAULT_SEED = 12345


def intrinsic(
    spot_grid: npt.NDArray[np.float64], strike: float, option_type: OptionType
) -> npt.NDArray[np.float64]:
    """Vectorized option intrinsic value on an array of spot prices."""
    if option_type == "C":
        return np.maximum(spot_grid - strike, 0.0)
    if option_type == "P":
        return np.maximum(strike - spot_grid, 0.0)
    raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")


def mean_stderr(values: npt.NDArray[np.float64], *, antithetic: bool) -> tuple[float, float]:
    """Sample mean and standard error of the mean.

    With antithetic sampling the draws come in perfectly negatively-correlated pairs
    (Z, -Z), so treating all 2m values as independent would understate the variance.
    We collapse each pair into its mean and compute the stderr across the m *pair
    means* — the statistically correct unit of independent replication.
    """
    values = np.asarray(values, dtype=float)
    if antithetic:
        m = values.size // 2
        est = 0.5 * (values[:m] + values[m : 2 * m])
    else:
        est = values
    n = est.size
    mean = float(est.mean())
    # ddof=1 (sample) variance; stderr of the mean = s / sqrt(n).
    stderr = float(est.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, stderr
