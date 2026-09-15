"""Non-negotiable runtime contract for the two-inner Davidson seam."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DavidsonShape:
    nb: int
    ng: int
    nproj: int
    batch: int

    @property
    def m(self) -> int:
        return 2 * self.nb

    def resident_bytes(self) -> dict[str, int]:
        k = 2 * self.ng
        return {
            "X_T": k * self.m * 8,
            "P_PT": self.nproj * self.m * 8,
            "residual": k * self.nb * 8,
            "RR_H_S": 2 * self.m * self.m * 8,
            "RR_eigenvectors_workspace_excluding_solver_query": self.m * self.m * 8,
        }


def refresh_dynamic_hamiltonian(descriptor, calc) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Read mutable ``vt/dH/dO`` from this live GPAW calculation."""
    from gpaw.utilities import unpack_hermitian

    wfs, ham, kpt = calc.wfs, calc.hamiltonian, calc.wfs.kpt_u[0]
    if descriptor.nb != kpt.psit.array.shape[0] or descriptor.nproj != sum(s.ni for s in wfs.setups):
        raise ValueError("live GPAW state no longer matches descriptor geometry")
    dH = np.zeros_like(descriptor.dH)
    dO = np.zeros_like(descriptor.dO)
    for atom, (start, end) in enumerate(zip(descriptor.paw.offsets, descriptor.paw.offsets[1:])):
        dH[start:end, start:end] = unpack_hermitian(
            np.ascontiguousarray(ham.dH_asp[atom][0])).real
        dO[start:end, start:end] = wfs.setups[atom].dO_ii.real
    volume = float(abs(np.linalg.det(wfs.gd.cell_cv)))
    return (np.ascontiguousarray(ham.vt_sG[kpt.s]).ravel(), dH, dO,
            2.0 * volume / descriptor.nr**2, descriptor.nr / volume)


def validate_graph(shape: DavidsonShape) -> dict[str, object]:
    if min(shape.nb, shape.ng, shape.nproj, shape.batch) <= 0:
        raise ValueError("all runtime Davidson dimensions must be positive")
    return {"m": shape.m, "rr_backend": "cusolverDnDsygvd", "itype": 1,
            "uplo": "lower", "inner_iterations": 2,
            "rotation": "same C rotates [X,T] and [P,PT]",
            "stale_HX_SX": False, "state_refresh": "vt,dH,dO per SCF iteration",
            "resident_bytes": shape.resident_bytes()}
