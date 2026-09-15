"""Descriptor-driven resident packed-Gamma regular Hamiltonian action."""
from __future__ import annotations

import ctypes as C

import numpy as np

from .. import runtime
from ..descriptor import GeneralDescriptor

CUFFT_D2Z, CUFFT_Z2D = 106, 108

SRC = r'''extern "C" {
__global__ void zero(double2* a, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)a[i]=make_double2(0.,0.); }
__global__ void embed(const double2* x,const long long* q,double2* a,int ng,int nr,int nq,int nb) {
 int g=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;
 if(g<ng&&b<nb) { double2 z=x[b*ng+g]; a[b*nq+q[g]]=make_double2(z.x/nr,z.y/nr); }
}
__global__ void mirror(double2* a,const long long* target,const long long* source,int count,int nq,int nb) {
 int i=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;
 if(i<count&&b<nb) { double2 z=a[b*nq+source[i]]; a[b*nq+target[i]]=make_double2(z.x,-z.y); }
}
__global__ void local(double* r,const double* v,int nr,int nb) { int i=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y; if(i<nr&&b<nb)r[b*nr+i]*=v[i]; }
__global__ void extract(const double2* a,const double2* x,const long long* q,const double* k,double2* y,int ng,int nq,int nb) {
 int g=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;
 if(g<ng&&b<nb){double2 u=a[b*nq+q[g]],z=x[b*ng+g];y[b*ng+g]=make_double2(u.x+k[g]*z.x,u.y+k[g]*z.y);}
}
}'''


class GammaHregGPU(runtime.GPU):
    """One persistent descriptor-bound FFT plan and all Hreg buffers."""

    def __init__(self, descriptor: GeneralDescriptor, batch: int = 32) -> None:
        super().__init__()
        self.driver = self.driver
        self.fft = runtime._load_configured("cufft")
        if self.module.value:
            self.driver.cuModuleUnload(self.module)
            self.module = runtime.P()
        self.descriptor = descriptor
        self.batch = min(batch, descriptor.nb)
        self.load_module(SRC, b"physop-hreg.cu")
        self._allocations: list[runtime.U64] = []
        self.dq, self.dv, self.dk, self.dtarget, self.dsource = (
            self._alloc(a.nbytes) for a in (descriptor.q_map, descriptor.potential,
                                             descriptor.kinetic, descriptor.gamma_mirror_target,
                                             descriptor.gamma_mirror_source))
        for dst, src in ((self.dq, descriptor.q_map), (self.dv, descriptor.potential),
                         (self.dk, descriptor.kinetic), (self.dtarget, descriptor.gamma_mirror_target),
                         (self.dsource, descriptor.gamma_mirror_source)):
            if src.nbytes:
                self.h2d(dst, src)
        nq = int(np.prod(descriptor.packed_shape))
        self.spectrum = self._alloc(self.batch * nq * 16)
        self.real = self._alloc(self.batch * descriptor.nr * 8)
        self._nq = nq
        self.r2c = self._plan(CUFFT_D2Z)
        self.c2r = self._plan(CUFFT_Z2D)
        self.fzero, self.fembed, self.fmirror, self.flocal, self.fextract = (
            self.fn(name) for name in ("zero", "embed", "mirror", "local", "extract"))

    def _alloc(self, nbytes: int) -> runtime.U64:
        device = self.alloc(max(1, nbytes))
        self._allocations.append(device)
        return device

    def _plan(self, kind: int) -> runtime.I:
        nx, ny, nz = self.descriptor.fft_shape
        nzp = nz // 2 + 1
        real_shape, packed_shape = (runtime.I * 3)(nx, ny, nz), (runtime.I * 3)(nx, ny, nzp)
        plan = runtime.I()
        if kind == CUFFT_D2Z:
            rc = self.fft.cufftPlanMany(C.byref(plan), 3, real_shape, real_shape, 1,
                                         self.descriptor.nr, packed_shape, 1, self._nq,
                                         kind, self.batch)
        else:
            rc = self.fft.cufftPlanMany(C.byref(plan), 3, real_shape, packed_shape, 1,
                                         self._nq, real_shape, 1, self.descriptor.nr,
                                         kind, self.batch)
        if rc:
            raise RuntimeError(f"cuFFT plan status {rc}")
        return plan

    def apply(self, x: runtime.U64, y: runtime.U64, columns: int) -> None:
        if not 0 < columns <= self.batch:
            raise ValueError("invalid resident Hreg tile width")
        d = self.descriptor
        self.launch(self.fzero, ((self._nq * columns + 255) // 256, 1),
                    [self.spectrum, self._nq * columns])
        self.launch(self.fembed, ((d.ng + 255) // 256, columns),
                    [x, self.dq, self.spectrum, d.ng, d.nr, self._nq, columns])
        if d.gamma_mirror_target.size:
            self.launch(self.fmirror, ((d.gamma_mirror_target.size + 255) // 256, columns),
                        [self.spectrum, self.dtarget, self.dsource,
                         d.gamma_mirror_target.size, self._nq, columns])
        if self.fft.cufftExecZ2D(self.c2r, runtime.P(self.spectrum.value),
                                 runtime.P(self.real.value)):
            raise RuntimeError("cuFFT C2R")
        self.launch(self.flocal, ((d.nr + 255) // 256, columns),
                    [self.real, self.dv, d.nr, columns])
        if self.fft.cufftExecD2Z(self.r2c, runtime.P(self.real.value),
                                 runtime.P(self.spectrum.value)):
            raise RuntimeError("cuFFT R2C")
        self.launch(self.fextract, ((d.ng + 255) // 256, columns),
                    [self.spectrum, x, self.dq, self.dk, y, d.ng, self._nq, columns])

    def close(self) -> None:
        if getattr(self, "r2c", runtime.I()).value:
            self.fft.cufftDestroy(self.r2c)
            self.r2c = runtime.I()
        if getattr(self, "c2r", runtime.I()).value:
            self.fft.cufftDestroy(self.c2r)
            self.c2r = runtime.I()
        for device in getattr(self, "_allocations", []):
            self.driver.cuMemFree_v2(device)
        self._allocations = []
        super().close()
