"""Resident Gamma H/S executor with PAW projector corrections."""
from __future__ import annotations

import ctypes as C

from .. import runtime
from ..descriptor import GeneralDescriptor
from .hreg import GammaHregGPU

BLAS = runtime._load_configured("cublas")
OP_N, OP_T = 0, 1


def _ck(code: int, where: str) -> None:
    if code:
        raise RuntimeError(f"{where}: cuBLAS status {code}")


class GeneralGammaExecutor(GammaHregGPU):
    """Hreg plus B/dH/B† and B/dO/B† for arbitrary packed-Gamma descriptors."""

    def __init__(self, descriptor: GeneralDescriptor, batch: int = 32) -> None:
        super().__init__(descriptor, batch)
        d = self.descriptor
        self.handle = runtime.P()
        _ck(BLAS.cublasCreate_v2(C.byref(self.handle)), "cublas create")
        self.dB_integrate = self._alloc(d.projector_integrate.nbytes)
        self.dB_add = self._alloc(d.projector_add.nbytes)
        self.ddH = self._alloc(d.dH.nbytes)
        self.ddO = self._alloc(d.dO.nbytes)
        self.p = self._alloc(self.batch * d.nproj * 8)
        self.q = self._alloc(self.batch * d.nproj * 8)
        self.hnl = self._alloc(self.batch * 2 * d.ng * 8)
        self.snl = self._alloc(self.batch * 2 * d.ng * 8)
        for target, source in ((self.dB_integrate, d.projector_integrate),
                               (self.dB_add, d.projector_add), (self.ddH, d.dH),
                               (self.ddO, d.dO)):
            self.h2d(target, source)
        self.counts = {"hreg": 0, "B": 0, "Bdag": 0, "PAW_dH": 0,
                       "PAW_dO": 0, "resident_h2d_inside": 0,
                       "resident_d2h_inside": 0}

    def _gemm(self, opa: int, m: int, n: int, k: int, alpha: float,
              a: runtime.U64, lda: int, b: runtime.U64, ldb: int, beta: float,
              c: runtime.U64, ldc: int, where: str) -> None:
        aa, bb = C.c_double(alpha), C.c_double(beta)
        _ck(BLAS.cublasDgemm_v2(self.handle, opa, OP_N, m, n, k, C.byref(aa),
                                runtime.P(a.value), lda, runtime.P(b.value), ldb,
                                C.byref(bb), runtime.P(c.value), ldc), where)

    def project(self, x: runtime.U64, out: runtime.U64, columns: int) -> None:
        d = self.descriptor
        self._gemm(OP_N, d.nproj, columns, 2 * d.ng, 2.0 / d.nr,
                   self.dB_integrate, d.nproj, x, 2 * d.ng, 0.0, out,
                   d.nproj, "B X")
        self.counts["B"] += 1

    def transform_paw(self, p: runtime.U64, block: runtime.U64, out: runtime.U64,
                      columns: int, label: str) -> None:
        d = self.descriptor
        self._gemm(OP_N, d.nproj, columns, d.nproj, 1.0, block, d.nproj,
                   p, d.nproj, 0.0, out, d.nproj, label)
        self.counts[label] += 1

    def _bdag_scaled(self, p: runtime.U64, out: runtime.U64, columns: int,
                     inverse_dv: float) -> None:
        d = self.descriptor
        self._gemm(OP_T, 2 * d.ng, columns, d.nproj, inverse_dv,
                   self.dB_add, d.nproj, p, d.nproj, 0.0, out, 2 * d.ng,
                   "Bdag P")
        self.counts["Bdag"] += 1

    def apply_hs(self, x: runtime.U64, h: runtime.U64, s: runtime.U64,
                 columns: int, inverse_dv: float) -> None:
        if not 0 < columns <= self.batch:
            raise ValueError("invalid Gamma executor tile width")
        d = self.descriptor
        self.apply(x, h, columns)
        self.counts["hreg"] += 1
        _ck(BLAS.cublasDcopy_v2(self.handle, 2 * d.ng * columns,
                                runtime.P(x.value), 1, runtime.P(s.value), 1),
            "S identity")
        self.project(x, self.p, columns)
        self.transform_paw(self.p, self.ddH, self.q, columns, "PAW_dH")
        self._bdag_scaled(self.q, self.hnl, columns, inverse_dv)
        self.transform_paw(self.p, self.ddO, self.q, columns, "PAW_dO")
        self._bdag_scaled(self.q, self.snl, columns, inverse_dv)
        one = C.c_double(1.0)
        for destination, correction, name in ((h, self.hnl, "Hnl"),
                                               (s, self.snl, "S")):
            _ck(BLAS.cublasDaxpy_v2(self.handle, 2 * d.ng * columns,
                                    C.byref(one), runtime.P(correction.value), 1,
                                    runtime.P(destination.value), 1),
                name + " accumulation")

    def close(self) -> None:
        if getattr(self, "handle", runtime.P()).value:
            BLAS.cublasDestroy_v2(self.handle)
            self.handle = runtime.P()
        super().close()
