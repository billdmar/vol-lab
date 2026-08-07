"""CRR binomial-tree pricing engine (European + American) and convergence harness."""

from src.lattice.crr import CRRBinomial, convergence_order

__all__ = ["CRRBinomial", "convergence_order"]
