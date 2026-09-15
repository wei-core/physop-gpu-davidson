#!/usr/bin/env python3
"""Tiny end-to-end numerical smoke for descriptor, H/S, projector and Davidson."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))


def _flatten(p_axi: dict[int, np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([p_axi[a] for a in sorted(p_axi)], axis=1))


def live_reference(calc, x: np.ndarray):
    from gpaw.utilities import unpack_hermitian
    wfs, ham, kpt = calc.wfs, calc.hamiltonian, calc.wfs.kpt_u[0]
    h = np.empty_like(x)
    wfs.apply_pseudo_hamiltonian(kpt, ham, x, h)
    p_axi = wfs.pt.integrate(x, q=kpt.q)
    hcorr = {a: p @ unpack_hermitian(np.ascontiguousarray(ham.dH_asp[a][0]))
             for a, p in p_axi.items()}
    s = x.copy()
    wfs.pt.add(h, hcorr, kpt.q)
    wfs.pt.add(s, {a: p @ wfs.setups[a].dO_ii for a, p in p_axi.items()}, kpt.q)
    return h, s, p_axi


def main() -> int:
    from physop_gpu.descriptor import from_gpaw
    from physop_gpu.materials.sic import make_calculator
    from physop_gpu.cuda.executor import GeneralGammaExecutor
    from physop_gpu.davidson.contract import refresh_dynamic_hamiltonian
    from physop_gpu.davidson.solver import GenericDavidson

    atoms, calc = make_calculator(None, "SIC-008", fixed_iterations=2)
    executor = None
    native_calc = None
    try:
        descriptor = from_gpaw("SIC-008", calc)
        rng = np.random.default_rng(20260915)
        columns = min(2, descriptor.nb)
        x_complex = np.ascontiguousarray(rng.normal(size=(columns, descriptor.ng)) +
                                          1j * rng.normal(size=(columns, descriptor.ng)))
        h_ref, s_ref, p_ref_axi = live_reference(calc, x_complex)
        p_ref = _flatten(p_ref_axi)
        inverse_dv = descriptor.nr / float(np.linalg.det(calc.wfs.gd.cell_cv))
        executor = GeneralGammaExecutor(descriptor, batch=columns)
        dx, dh, ds, dp = (executor._alloc(a.nbytes) for a in
                          (x_complex, np.empty_like(x_complex), np.empty_like(x_complex), p_ref))
        executor.h2d(dx, x_complex)
        executor.apply_hs(dx, dh, ds, columns, inverse_dv)
        executor.project(dx, dp, columns)
        executor.driver.cuCtxSynchronize()
        h_gpu, s_gpu, p_gpu = np.empty_like(x_complex), np.empty_like(x_complex), np.empty_like(p_ref)
        executor.d2h(h_gpu, dh); executor.d2h(s_gpu, ds); executor.d2h(p_gpu, dp)
        executor.driver.cuCtxSynchronize()
        hs_error = relative(h_gpu, h_ref)
        s_error = relative(s_gpu, s_ref)
        b_error = relative(p_gpu, p_ref)

        # Validate the inverse projector independently against GPAW's PWLFC
        # add() path.  The host input is the same real Nproj projection layout
        # used by the production Davidson state.
        p_test = np.ascontiguousarray(rng.normal(size=(columns, descriptor.nproj)))
        d_p_test = executor._alloc(p_test.nbytes)
        bdagger_gpu = np.empty_like(x_complex)
        d_bdagger = executor._alloc(bdagger_gpu.nbytes)
        executor.h2d(d_p_test, p_test)
        executor._bdag_scaled(d_p_test, d_bdagger, columns, inverse_dv)
        executor.driver.cuCtxSynchronize(); executor.d2h(bdagger_gpu, d_bdagger)
        executor.driver.cuCtxSynchronize()
        p_blocks = {atom: p_test[:, start:end]
                    for atom, (start, end) in enumerate(zip(descriptor.paw.offsets,
                                                               descriptor.paw.offsets[1:]))}
        bdagger_ref = np.zeros_like(x_complex)
        calc.wfs.pt.add(bdagger_ref, p_blocks, calc.wfs.kpt_u[0].q)
        bdagger_error = relative(bdagger_gpu, bdagger_ref)

        wfs, kpt = calc.wfs, calc.wfs.kpt_u[0]
        p_axi = wfs.pt.integrate(kpt.psit.array, q=kpt.q)
        kpt.projections.array[:] = _flatten(p_axi)
        wfs.eigensolver.initialize(wfs)
        wfs.eigensolver.subspace_diagonalize(calc.hamiltonian, wfs, kpt)
        x_state = np.ascontiguousarray(kpt.psit.array)
        p_state = np.ascontiguousarray(kpt.projections.array)
        eps_state = np.ascontiguousarray(kpt.eps_n)
        vt, dh_live, do_live, alpha, inverse_dv = refresh_dynamic_hamiltonian(descriptor, calc)
        davidson = GenericDavidson(descriptor, alpha, inverse_dv)
        davidson.update_dynamic(vt, dh_live, do_live)
        out = davidson.run_two_inner(x_state, p_state, eps_state)
        davidson.h2d(davidson.x, out["x"])
        davidson.project(davidson.x, davidson.pt, descriptor.nb)
        davidson.driver.cuCtxSynchronize()
        p_after = np.empty_like(out["p"]); davidson.d2h(p_after, davidson.pt)
        davidson.driver.cuCtxSynchronize()
        p_identity_error = relative(out["p"], p_after)
        # Independent native trajectory is the GPAW authority for the Ritz metric.
        # Keep the smoke small and do not run SIC-216 here.
        from physop_gpu.materials.sic import make_calculator as make_native
        _, native_calc = make_native(None, "SIC-008", fixed_iterations=2)
        nwfs, nkpt = native_calc.wfs, native_calc.wfs.kpt_u[0]
        np_axi = nwfs.pt.integrate(nkpt.psit.array, q=nkpt.q)
        nkpt.projections.array[:] = _flatten(np_axi)
        nwfs.eigensolver.initialize(nwfs)
        nwfs.eigensolver.iterate_one_k_point(native_calc.hamiltonian, nwfs, nkpt, np.ones(descriptor.nb))
        ritz_error = relative(out["eps"], nkpt.eps_n)
        result = {"status": "PASS" if max(hs_error, s_error, b_error, bdagger_error, p_identity_error, ritz_error) < 5e-10 else "FAIL",
                  "case": "SIC-008", "shape": descriptor.metadata(),
                  "H error": hs_error, "S error": s_error, "B error": b_error,
                  "Bdagger error": bdagger_error, "Ritz error": ritz_error,
                  "P=B(X) error": p_identity_error, "inner_iterations": 2}
        print(json.dumps(result, indent=2, allow_nan=True))
        return 0 if result["status"] == "PASS" else 2
    finally:
        if native_calc is not None: native_calc.close()
        if executor is not None: executor.close()
        if 'davidson' in locals(): davidson.close()
        calc.close()


if __name__ == "__main__":
    raise SystemExit(main())
