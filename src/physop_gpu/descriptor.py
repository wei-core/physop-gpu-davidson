"""Runtime-owned descriptor extracted from a live GPAW Gamma calculation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

REPRESENTATION_GAMMA_PACKED = "GAMMA_PACKED"
REPRESENTATION_GENERAL_COMPLEX = "GENERAL_COMPLEX"


@dataclass(frozen=True)
class PawLayout:
    ni: tuple[int, ...]
    offsets: tuple[int, ...]

    @property
    def nproj(self) -> int:
        return self.offsets[-1]


@dataclass
class GeneralDescriptor:
    """Case-dependent binding for the resident Gamma executor."""

    case: str
    representation: str
    n_atom: int
    nb: int
    ng: int
    nr: int
    fft_shape: tuple[int, int, int]
    q_map: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    projector_integrate: np.ndarray
    projector_add: np.ndarray
    dH: np.ndarray
    dO: np.ndarray
    paw: PawLayout
    gamma_self_conjugate: np.ndarray
    gamma_mirror_target: np.ndarray
    gamma_mirror_source: np.ndarray

    @property
    def nproj(self) -> int:
        return self.paw.nproj

    @property
    def packed_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.fft_shape
        return nx, ny, nz // 2 + 1

    def metadata(self) -> dict[str, Any]:
        value = asdict(self.paw)
        return {
            "case": self.case,
            "representation": self.representation,
            "N_atom": self.n_atom,
            "Nb": self.nb,
            "NG": self.ng,
            "NR": self.nr,
            "Nproj": self.nproj,
            "FFT_shape": list(self.fft_shape),
            "packed_shape": list(self.packed_shape),
            "Q_map_length": int(self.q_map.size),
            "projector_shape": list(self.projector_integrate.shape),
            "ni": value["ni"],
            "PAW_offsets": value["offsets"],
            "gamma_self_conjugate_count": int(self.gamma_self_conjugate.size),
            "gamma_mirror_pair_count": int(self.gamma_mirror_target.size),
            "dtype": str(self.projector_integrate.dtype),
        }

    def validate(self) -> dict[str, Any]:
        nx, ny, nz = self.fft_shape
        nq = nx * ny * (nz // 2 + 1)
        errors: list[str] = []
        if self.representation != REPRESENTATION_GAMMA_PACKED:
            errors.append("Gamma executor received a non-Gamma descriptor")
        if self.q_map.shape != (self.ng,):
            errors.append("Q-map length does not match NG")
        if self.q_map.size and (self.q_map.min() < 0 or self.q_map.max() >= nq):
            errors.append("Q-map is outside packed FFT range")
        if np.unique(self.q_map).size != self.q_map.size:
            errors.append("Q-map is not injective")
        if self.nr != nx * ny * nz:
            errors.append("FFT volume does not match NR")
        if self.projector_integrate.shape != (2 * self.ng, self.nproj):
            errors.append("projector shape is not packed-Gamma K by Nproj")
        if self.projector_add.shape != (2 * self.ng, self.nproj):
            errors.append("projector add shape is not packed-Gamma K by Nproj")
        if self.dH.shape != (self.nproj, self.nproj):
            errors.append("dH shape does not match ragged PAW layout")
        if self.dO.shape != (self.nproj, self.nproj):
            errors.append("dO shape does not match ragged PAW layout")
        if len(self.paw.ni) != self.n_atom or len(self.paw.offsets) != self.n_atom + 1:
            errors.append("PAW atom layout is inconsistent")
        if self.paw.offsets[0] != 0 or self.paw.offsets[-1] != self.nproj:
            errors.append("PAW prefix offsets do not span Nproj")
        if any(b < a for a, b in zip(self.paw.offsets, self.paw.offsets[1:])):
            errors.append("PAW offsets are not monotone")
        return {"status": "PASS" if not errors else "FAIL", "errors": errors,
                **self.metadata()}


def _ragged_paw_blocks(wfs: Any, ham: Any) -> tuple[PawLayout, np.ndarray, np.ndarray]:
    from gpaw.utilities import unpack_hermitian

    ni = tuple(int(setup.ni) for setup in wfs.setups)
    offsets = [0]
    for count in ni:
        offsets.append(offsets[-1] + count)
    nproj = offsets[-1]
    dH = np.zeros((nproj, nproj), dtype=float)
    dO = np.zeros_like(dH)
    for atom, (start, end) in enumerate(zip(offsets, offsets[1:])):
        h = np.asarray(unpack_hermitian(np.ascontiguousarray(ham.dH_asp[atom][0])).real)
        o = np.asarray(wfs.setups[atom].dO_ii.real)
        if h.shape != (end - start, end - start) or o.shape != h.shape:
            raise ValueError(f"atom {atom}: GPAW PAW block shape disagrees with ni")
        dH[start:end, start:end] = h
        dO[start:end, start:end] = o
    return PawLayout(ni, tuple(offsets)), dH, dO


def _gamma_completion(q_map: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce conjugate-completion pairs for GPAW's half-packed FFT layout."""
    nx, ny, nz = shape
    nzp = nz // 2 + 1
    present = {int(q) for q in q_map}
    targets: list[int] = []
    sources: list[int] = []
    self_conjugate: list[int] = []
    for source in q_map:
        source = int(source)
        ix = source // (ny * nzp)
        iy = (source // nzp) % ny
        iz = source % nzp
        target = ((-ix) % nx * ny + (-iy) % ny) * nzp + iz
        if target == source:
            self_conjugate.append(source)
        elif target not in present:
            targets.append(target)
            sources.append(source)
    return (np.asarray(self_conjugate, dtype=np.int64),
            np.asarray(targets, dtype=np.int64), np.asarray(sources, dtype=np.int64))


def from_gpaw(case: str, calc: Any) -> GeneralDescriptor:
    """Extract a descriptor exclusively from the initialized GPAW calculation."""
    wfs = calc.wfs
    kpt = wfs.kpt_u[0]
    if kpt.q != 0 or wfs.pd.dtype is not float:
        raise ValueError("portable executor supports GPAW's packed Gamma representation only")
    pd = wfs.pd
    shape = tuple(int(x) for x in wfs.gd.N_c)
    q_map = np.ascontiguousarray(pd.Q_qG[kpt.q], dtype=np.int64)
    projector_add = np.ascontiguousarray(wfs.pt.expand(q=kpt.q), dtype=np.float64)
    projector_integrate = projector_add.copy()
    # GPAW's packed-Gamma integrate() applies the G=0 half weight; add()
    # leaves the reconstruction tile unweighted.  Both are explicit here.
    projector_integrate[0] *= 0.5
    paw, dH, dO = _ragged_paw_blocks(wfs, calc.hamiltonian)
    self_conjugate, targets, sources = _gamma_completion(q_map, shape)
    descriptor = GeneralDescriptor(
        case=case,
        representation=REPRESENTATION_GAMMA_PACKED,
        n_atom=len(wfs.setups),
        nb=int(kpt.psit.array.shape[0]),
        ng=int(q_map.size),
        nr=int(np.prod(shape)),
        fft_shape=shape,
        q_map=q_map,
        kinetic=np.ascontiguousarray(0.5 * pd.G2_qG[kpt.q], dtype=np.float64),
        potential=np.ascontiguousarray(calc.hamiltonian.vt_sG[kpt.s], dtype=np.float64).ravel(),
        projector_integrate=projector_integrate,
        projector_add=projector_add,
        dH=np.ascontiguousarray(dH),
        dO=np.ascontiguousarray(dO),
        paw=paw,
        gamma_self_conjugate=self_conjugate,
        gamma_mirror_target=targets,
        gamma_mirror_source=sources,
    )
    checked = descriptor.validate()
    if checked["status"] != "PASS":
        raise ValueError("invalid GPAW descriptor: " + "; ".join(checked["errors"]))
    return descriptor
