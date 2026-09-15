"""Frozen 3C-SiC geometry and runtime-derived GPAW case construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

CASES: dict[str, tuple[int, int, int]] = {
    "SIC-008": (1, 1, 1),
    "SIC-032": (2, 2, 1),
    "SIC-064": (2, 2, 2),
    "SIC-128": (2, 2, 4),
    "SIC-216": (3, 3, 3),
}


def case_config(case: str) -> dict[str, Any]:
    key = case.upper()
    if key not in CASES:
        raise ValueError(f"unknown case {case!r}; choose one of {', '.join(CASES)}")
    replication = CASES[key]
    n_atom = 8 * replication[0] * replication[1] * replication[2]
    nbands = int((1.20 * 4 * n_atom / 2) + 0.999999999)
    return {"case": key, "material": "3C-SiC", "replication": replication,
            "ecut_eV": 400, "nbands": nbands, "gamma": True,
            "xc": "LDA", "occupations_eV": 0.001,
            "nbands_policy": "ceil(1.20 * 4 * N_atom / 2)"}


def build_atoms(case: str | int):
    from ase.build import bulk

    if isinstance(case, str):
        replication = CASES[case.upper()]
    else:
        replication = (int(case),) * 3
    return bulk("SiC", "zincblende", a=4.332, cubic=True).repeat(replication)


def make_calculator(txt: str | Path | None, case: str, *, fixed_iterations: int | None = None,
                    maxiter: int | None = None, nbands: int | None = None):
    """Construct the GPAW authority used by E3/E4/E5.

    The descriptor later reads NG/NR/Nproj/FFT dimensions from the initialized
    calculation; only geometry, energy cutoff and band policy are case inputs.
    """
    from gpaw import GPAW, PW
    from gpaw.eigensolvers.davidson import Davidson
    from gpaw.occupations import FermiDirac

    config = case_config(case)
    replication = config["replication"]
    atoms = build_atoms(case)
    nbands = int(nbands or config["nbands"])
    if fixed_iterations is not None:
        convergence = {"density": 1.0e-8, "bands": -5,
                       "maximum iterations": int(fixed_iterations)}
        maxiter = int(fixed_iterations)
    else:
        convergence = {"density": 1.0e-8, "bands": -5}
        maxiter = int(maxiter or 333)
    calc = GPAW(mode=PW(config["ecut_eV"]), xc=config["xc"],
                kpts={"size": (1, 1, 1), "gamma": True},
                occupations=FermiDirac(config["occupations_eV"]), nbands=nbands,
                spinpol=False, charge=0, symmetry="off", maxiter=maxiter,
                convergence=convergence, eigensolver=Davidson(niter=2),
                setups="paw", txt=None if txt is None else str(txt))
    atoms.calc = calc
    calc.initialize(atoms)
    calc.set_positions(atoms)
    return atoms, calc
