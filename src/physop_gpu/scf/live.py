"""E4 live seam and E5 fixed/converged run helpers."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..descriptor import from_gpaw
from ..davidson.contract import refresh_dynamic_hamiltonian
from ..davidson.solver import GenericDavidson
from ..materials.sic import make_calculator
from ..runtime import check

TOLERANCE = 5.0e-11


def require_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GPAW_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for reproducible BLAS/GPAW execution")


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def state_record(calc: Any) -> dict[str, Any]:
    wfs, ham, kpt = calc.wfs, calc.hamiltonian, calc.wfs.kpt_u[0]
    return {"iteration": int(calc.scf.niter), "free_energy_ha": float(ham.e_total_free),
            "eps": np.ascontiguousarray(kpt.eps_n.copy()),
            "eigensolver_error": float(wfs.eigensolver.error),
            "potential": np.ascontiguousarray(ham.vt_sG[kpt.s].copy()),
            "density": np.ascontiguousarray(calc.density.nt_sG.copy())}


def shape_metadata(calc: Any, case: str, nbands: int | None = None) -> dict[str, Any]:
    wfs, kpt = calc.wfs, calc.wfs.kpt_u[0]
    fft_shape = tuple(int(value) for value in wfs.gd.N_c)
    return {"case": case, "representation": "GAMMA_PACKED", "N_atom": len(wfs.setups),
            "Nb": int(nbands or kpt.psit.array.shape[0]), "NG": int(wfs.pd.Q_qG[kpt.q].size),
            "NR": int(np.prod(fft_shape)), "Nproj": int(sum(setup.ni for setup in wfs.setups)),
            "FFT_shape": list(fft_shape)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class LiveGenericAdapter:
    """Replace only GPAW's iterate_one_k_point at the native SCF seam."""

    def __init__(self, calc: Any, output: Path | None, case: str):
        setup_started = time.perf_counter()
        self.case, self.calc = case, calc
        self.output = output
        self.iteration = 0
        self.records: list[dict[str, Any]] = []
        self.descriptor = from_gpaw(case, calc)
        vt, dH, dO, alpha, inverse_dv = refresh_dynamic_hamiltonian(self.descriptor, calc)
        self.gpu = GenericDavidson(self.descriptor, alpha, inverse_dv)
        self.gpu.update_dynamic(vt, dH, dO)
        check(self.gpu.driver.cuCtxSynchronize(), "initial dynamic state")
        self.cold_setup_s = time.perf_counter() - setup_started
        self._previous_dynamic_digest: str | None = None

    def replace(self, solver: Any, ham: Any, wfs: Any, kpt: Any,
                weights: np.ndarray) -> float:
        total_started = time.perf_counter()
        solver.subspace_diagonalize(ham, wfs, kpt)
        x = np.ascontiguousarray(kpt.psit.array.copy())
        p = np.ascontiguousarray(kpt.projections.array.copy())
        eps = np.ascontiguousarray(kpt.eps_n.copy())
        live_vt, live_dH, live_dO, _, _ = refresh_dynamic_hamiltonian(self.descriptor, self.calc)
        dynamic_digest = _digest(live_vt, live_dH, live_dO)
        changed = self._previous_dynamic_digest is None or dynamic_digest != self._previous_dynamic_digest
        self._previous_dynamic_digest = dynamic_digest
        volume = float(abs(np.linalg.det(wfs.gd.cell_cv)))
        alpha = 2.0 * volume / self.descriptor.nr**2
        inverse_dv = self.descriptor.nr / volume
        self.gpu.alpha, self.gpu.inverse_dv = alpha, inverse_dv
        self.gpu.update_dynamic(live_vt, live_dH, live_dO)
        check(self.gpu.driver.cuCtxSynchronize(), "dynamic refresh")
        self.gpu.h2d(self.gpu.x, x); self.gpu.h2d(self.gpu.px, p); self.gpu.h2d(self.gpu.eps, eps)
        check(self.gpu.driver.cuCtxSynchronize(), "live state H2D")
        self.gpu.timing = {}
        first_started = time.perf_counter(); self.gpu.inner(); first_s = time.perf_counter() - first_started
        second_started = time.perf_counter(); self.gpu.residual_to_t()
        check(self.gpu.driver.cuCtxSynchronize(), "inner2 residual")
        residual = np.empty_like(x); residual_d2h_started = time.perf_counter()
        self.gpu.d2h(residual, self.gpu.r); check(self.gpu.driver.cuCtxSynchronize(), "inner2 residual D2H")
        residual_d2h_s = time.perf_counter() - residual_d2h_started
        error = float(np.dot(weights, [np.real(wfs.integrate(row, row, global_integral=False))
                                       for row in residual]))
        second_residual_s = time.perf_counter() - second_started
        finish_started = time.perf_counter(); self.gpu.project_t(); self.gpu.projected()
        self.gpu.rr_solve(); self.gpu.rotate(); check(self.gpu.driver.cuCtxSynchronize(), "two-inner completion")
        second_finish_s = time.perf_counter() - finish_started
        xo, po, eo = np.empty_like(x), np.empty_like(p), np.empty_like(eps)
        state_d2h_started = time.perf_counter(); self.gpu.d2h(xo, self.gpu.x); self.gpu.d2h(po, self.gpu.px); self.gpu.d2h(eo, self.gpu.eps)
        check(self.gpu.driver.cuCtxSynchronize(), "live state D2H")
        state_d2h_s = time.perf_counter() - state_d2h_started
        kpt.psit.array[...] = xo; kpt.projections.array[...] = po; kpt.eps_n[...] = eo
        total_s = time.perf_counter() - total_started
        self.records.append({"iteration": self.iteration + 1, "dynamic_state_digest": dynamic_digest,
                             "dynamic_changed_from_previous": changed,
                             "dynamic_refresh": {"vt": True, "dH": True, "dO": True},
                             "alpha": alpha, "inverse_dv": inverse_dv, "inner_iterations": 2,
                             "inner1_s": first_s, "inner2_residual_s": second_residual_s,
                             "inner2_residual_d2h_s": residual_d2h_s, "inner2_finish_s": second_finish_s,
                             "state_d2h_s": state_d2h_s, "gpu_davidson_s": first_s + second_residual_s + second_finish_s,
                             "total_replacement_s": total_s, "reported_error": error,
                             "gpu_stage_s": {str(k): float(v) for k, v in self.gpu.timing.items()}})
        self.iteration += 1
        return error

    def close(self) -> None:
        self.gpu.close()


def run_case(case: str, mode: str, backend: str, output: Path | None = None) -> dict[str, Any]:
    """Run one fixed5 or converged case through native or GPU GPAW seam."""
    require_threads()
    case = case.upper()
    fixed = mode.lower() == "fixed5"
    if mode.lower() not in {"fixed5", "converged"}:
        raise ValueError("mode must be fixed5 or converged")
    if backend not in {"native", "gpu"}:
        raise ValueError("backend must be native or gpu")
    run_dir = Path(output or Path.cwd() / "runs" / case.lower() / f"{mode}-{backend}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms, calc = make_calculator(run_dir / "gpaw.txt", case,
                                  fixed_iterations=5 if fixed else None)
    solver, wfs = calc.wfs.eigensolver, calc.wfs
    solver.keep_htpsit = False
    adapter = LiveGenericAdapter(calc, run_dir, case) if backend == "gpu" else None
    original = solver.iterate_one_k_point
    native_times: list[float] = []
    if adapter is not None:
        solver.iterate_one_k_point = lambda ham, wf, kpt, weights: adapter.replace(solver, ham, wf, kpt, weights)
    else:
        def timed(ham, wf, kpt, weights):
            started = time.perf_counter(); value = original(ham, wf, kpt, weights)
            native_times.append(time.perf_counter() - started); return value
        solver.iterate_one_k_point = timed
    from gpaw.scf import SCFLoop
    states: list[dict[str, Any]] = []
    old_check = SCFLoop.check_convergence
    def checked(scf, *args, **kwargs):
        value = old_check(scf, *args, **kwargs)
        if scf is calc.scf: states.append(state_record(calc))
        return value
    SCFLoop.check_convergence = checked
    started = time.perf_counter(); terminal, exception, energy = "CONVERGED", None, None
    try:
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        from gpaw import KohnShamConvergenceError
        if isinstance(exc, KohnShamConvergenceError): terminal = "FIXED_ITERATION_LIMIT"
        else: terminal, exception = "ERROR", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    SCFLoop.check_convergence = old_check; solver.iterate_one_k_point = original
    if fixed and terminal == "CONVERGED" and int(calc.scf.niter) >= 5:
        terminal = "FIXED_ITERATION_LIMIT"
    records = [] if adapter is None else adapter.records
    result = {"schema": "physop-portable-scf-v1", "case": case, "material": "3C-SiC",
              "mode": mode, "backend": backend, "terminal": terminal, "exception": exception,
              "scf_iterations": int(calc.scf.niter), "shape": shape_metadata(calc, case),
              "davidson_invocations": len(records) if adapter else len(native_times),
              "end_to_end_s": elapsed, "davidson_total_s": sum(r["total_replacement_s"] for r in records) if adapter else sum(native_times),
              "energy_ev": energy, "trajectory": [{k: _jsonable(v) for k, v in item.items()} for item in states],
              "adapter_records": records}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=_jsonable) + "\n")
    if adapter is not None: adapter.close()
    calc.close()
    return result
