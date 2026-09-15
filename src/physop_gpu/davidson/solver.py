"""Real packed-Gamma two-inner Davidson with resident CUDA state."""
from __future__ import annotations

import ctypes as C
import time

import numpy as np

from .. import runtime
from ..cuda.executor import BLAS, OP_N, OP_T, GeneralGammaExecutor, _ck
from ..cuda.solver import (CUBLAS_FILL_MODE_LOWER, CUSOLVER_EIG_MODE_VECTOR,
                           CUSOLVER_EIG_TYPE_1, SOLVER)


def at(base: runtime.U64, byte_offset: int) -> runtime.U64:
    return runtime.U64(base.value + byte_offset)


def d2d(dst: runtime.U64, src: runtime.U64, nbytes: int, driver) -> None:
    runtime.check(driver.cuMemcpyDtoD_v2(dst, src, nbytes), "E4 device copy")


def _source(nb: int, ng: int, nproj: int, alpha: float) -> str:
    return rf'''extern "C" {{
__global__ void residual0(double2*r,const double2*x,const double*e){{int g=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(g<{ng}){{int i=b*{ng}+g;r[i].x-=e[b]*x[i].x;r[i].y-=e[b]*x[i].y;}}}}
__global__ void ekin(const double2*x,const double*k,double*e){{__shared__ double q[256];int b=blockIdx.x,t=threadIdx.x;double v=0.;for(int g=t;g<{ng};g+=256){{double2 z=x[b*{ng}+g];v+=k[g]*(z.x*z.x+z.y*z.y);}}q[t]=v;__syncthreads();for(int n=128;n;n>>=1){{if(t<n)q[t]+=q[t+n];__syncthreads();}}if(!t)e[b]={alpha:.17g}*q[0];}}
__global__ void precond(const double2*r,const double*k,const double*e,double2*t){{int g=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(g<{ng}){{double x=2.*k[g]/(3.*e[b]);double a=27.+x*(18.+x*(12.+8.*x));double f=-4.*a/(3.*e[b]*(a+16.*x*x*x*x));int i=b*{ng}+g;t[i]=make_double2(f*r[i].x,f*r[i].y);}}}}
__global__ void combine(double*h,const double*s,const double*e){{int i=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(i<{nproj})h[b*{nproj}+i]-=e[b]*s[b*{nproj}+i];}}
__global__ void g0(double*c,const double*a,const double*b,int na,int nb){{int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(i<na&&j<nb)c[i+j*na]-=.5*{alpha:.17g}*a[i*{2*ng}]*b[j*{2*ng}];}}
__global__ void top(double*h,double*s,const double*e){{int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(i<{nb}&&j<{nb}){{h[i+j*{2*nb}]=(i==j?e[i]:0.);s[i+j*{2*nb}]=(i==j?1.:0.);}}}}
__global__ void store(double*dst,const double*src,int kind){{int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(i<{nb}&&j<{nb}){{int m={2*nb};if(kind==0){{if(i>=j)dst[({nb}+i)+({nb}+j)*m]=src[i+j*{nb}];}}else dst[({nb}+i)+j*m]=src[i+j*{nb}];}}}}
}}'''


class GenericDavidson(GeneralGammaExecutor):
    """Production E4 graph over an E3 descriptor-bound resident operator."""

    def __init__(self, descriptor, alpha: float, inverse_dv: float) -> None:
        super().__init__(descriptor, batch=min(32, descriptor.nb))
        self.nb, self.ng, self.nproj = descriptor.nb, descriptor.ng, descriptor.nproj
        self.m, self.k, self.alpha, self.inverse_dv = 2 * self.nb, 2 * self.ng, alpha, inverse_dv
        # Keep the E3 Hreg module alive while loading the separate E4 kernel
        # module; both modules' function handles remain resident.
        self.hreg_module, self.hreg_blob = self.module, self._blob
        self.module, self._blob = runtime.P(), None
        self.load_module(_source(self.nb, self.ng, self.nproj, alpha), b"physop-davidson.cu")
        self.e4module, self.e4_blob = self.module, self._blob
        self.fn4 = {name: self.fn(name) for name in
                    ("residual0", "ekin", "precond", "combine", "g0", "top", "store")}
        self.extra: list[runtime.U64] = []

        def alloc(nbytes: int) -> runtime.U64:
            device = self._alloc(nbytes)
            self.extra.append(device)
            return device

        self.z, self.pz = alloc(self.k * self.m * 8), alloc(self.nproj * self.m * 8)
        self.r, self.p3, self.mwork = alloc(self.k * self.nb * 8), alloc(self.nproj * self.nb * 8), alloc(self.nb * self.nb * 8)
        self.hmat, self.smat = alloc(self.m * self.m * 8), alloc(self.m * self.m * 8)
        self.eps, self.ekin_d, self.w, self.info = alloc(self.nb * 8), alloc(self.nb * 8), alloc(self.m * 8), alloc(4)
        self.solver = runtime.P()
        runtime.check(SOLVER.cusolverDnCreate(C.byref(self.solver)), "Dsygvd create")
        lw = runtime.I()
        runtime.check(SOLVER.cusolverDnDsygvd_bufferSize(
            self.solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, self.m, runtime.P(self.hmat.value), self.m,
            runtime.P(self.smat.value), self.m, runtime.P(self.w.value), C.byref(lw)),
            "Dsygvd workspace")
        self.lwork = lw.value
        self.work = alloc(self.lwork * 8)
        self.timing: dict[str, float] = {}

    @property
    def x(self): return self.z
    @property
    def t(self): return at(self.z, self.k * self.nb * 8)
    @property
    def px(self): return self.pz
    @property
    def pt(self): return at(self.pz, self.nproj * self.nb * 8)

    def launch4(self, name: str, grid: tuple[int, int], values: list) -> None:
        args, keep = [], []
        for value in values:
            converted = runtime.U64(value.value) if hasattr(value, "value") else runtime.I(value)
            keep.append(converted)
            args.append(C.cast(C.byref(converted), runtime.P))
        argv = (runtime.P * len(args))(*args)
        runtime.check(self.driver.cuLaunchKernel(self.fn4[name], grid[0], grid[1], 1,
                                                 256, 1, 1, 0, None, argv, None), name)

    def stage(self, name: str, call) -> None:
        started = time.perf_counter()
        call()
        runtime.check(self.driver.cuCtxSynchronize(), name + " sync")
        self.timing[name] = self.timing.get(name, 0.0) + time.perf_counter() - started

    def update_dynamic(self, vt, dh, do) -> None:
        self.h2d(self.dv, vt)
        self.h2d(self.ddH, dh)
        self.h2d(self.ddO, do)

    def hreg(self, src: runtime.U64, dst: runtime.U64) -> None:
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            self.apply(at(src, first * self.k * 8), at(dst, first * self.k * 8), n)

    def paw(self, left: runtime.U64, right: runtime.U64, beta: float) -> None:
        self._gemm(OP_T, self.nb, self.nb, self.nproj, 1.0, left, self.nproj,
                   right, self.nproj, beta, self.mwork, self.nb, "PAW Gram")

    def metric(self, a: runtime.U64, b: runtime.U64) -> None:
        self._gemm(OP_T, self.nb, self.nb, self.k, self.alpha, a, self.k,
                   b, self.k, 0.0, self.mwork, self.nb, "real Gamma Gram")
        self.launch4("g0", ((self.nb + 255) // 256, self.nb),
                     [self.mwork, a, b, self.nb, self.nb])

    def scatter(self, coeff: runtime.U64, out: runtime.U64) -> None:
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            self._bdag_scaled(at(coeff, first * self.nproj * 8), self.hnl, n, self.inverse_dv)
            one = C.c_double(1.0)
            _ck(BLAS.cublasDaxpy_v2(self.handle, self.k * n, C.byref(one),
                                    runtime.P(self.hnl.value), 1,
                                    runtime.P(at(out, first * self.k * 8).value), 1),
                "Bdag accumulation")

    def residual_to_t(self) -> None:
        self.hreg(self.x, self.r)
        self.launch4("residual0", ((self.ng + 255) // 256, self.nb),
                     [self.r, self.x, self.eps])
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            po = first * self.nproj * 8
            self.transform_paw(at(self.px, po), self.ddH, at(self.p3, po), n, "PAW_dH")
            self.transform_paw(at(self.px, po), self.ddO, at(self.pt, po), n, "PAW_dO")
        self.launch4("combine", ((self.nproj + 255) // 256, self.nb),
                     [self.p3, self.pt, self.eps])
        self.scatter(self.p3, self.r)
        self.launch4("ekin", (self.nb, 1), [self.x, self.dk, self.ekin_d])
        self.launch4("precond", ((self.ng + 255) // 256, self.nb),
                     [self.r, self.dk, self.ekin_d, self.t])

    def project_t(self) -> None:
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            self.project(at(self.t, first * self.k * 8), at(self.pt, first * self.nproj * 8), n)

    def projected(self) -> None:
        self.launch4("top", ((self.nb + 255) // 256, self.nb),
                     [self.hmat, self.smat, self.eps])
        self.hreg(self.t, self.r)
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            self.transform_paw(at(self.pt, first * self.nproj * 8), self.ddH,
                               at(self.p3, first * self.nproj * 8), n, "PAW_dH")
        self.metric(self.t, self.r)
        self.paw(self.pt, self.p3, 1.0)
        self.launch4("store", ((self.nb + 255) // 256, self.nb), [self.hmat, self.mwork, 0])
        self.metric(self.r, self.x)
        self.paw(self.p3, self.px, 1.0)
        self.launch4("store", ((self.nb + 255) // 256, self.nb), [self.hmat, self.mwork, 1])
        for first in range(0, self.nb, self.batch):
            n = min(self.batch, self.nb - first)
            self.transform_paw(at(self.pt, first * self.nproj * 8), self.ddO,
                               at(self.p3, first * self.nproj * 8), n, "PAW_dO")
        self.metric(self.t, self.t)
        self.paw(self.pt, self.p3, 1.0)
        self.launch4("store", ((self.nb + 255) // 256, self.nb), [self.smat, self.mwork, 0])
        self.metric(self.t, self.x)
        self.paw(self.p3, self.px, 1.0)
        self.launch4("store", ((self.nb + 255) // 256, self.nb), [self.smat, self.mwork, 1])

    def rr_solve(self) -> None:
        runtime.check(SOLVER.cusolverDnDsygvd(
            self.solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, self.m, runtime.P(self.hmat.value), self.m,
            runtime.P(self.smat.value), self.m, runtime.P(self.w.value),
            runtime.P(self.work.value), self.lwork, runtime.P(self.info.value)), "Dsygvd")
        runtime.check(self.driver.cuCtxSynchronize(), "Dsygvd sync")
        info = np.empty(1, np.int32)
        self.d2h(info, self.info)
        if info[0] != 0:
            raise RuntimeError(f"Dsygvd devInfo={info[0]}")

    def rotate(self) -> None:
        d2d(self.eps, self.w, self.nb * 8, self.driver)
        self._gemm(OP_N, self.k, self.nb, self.m, 1.0, self.z, self.k,
                   self.hmat, self.m, 0.0, self.r, self.k, "X rotation")
        self._gemm(OP_N, self.nproj, self.nb, self.m, 1.0, self.pz, self.nproj,
                   self.hmat, self.m, 0.0, self.p3, self.nproj, "P rotation")
        d2d(self.x, self.r, self.k * self.nb * 8, self.driver)
        d2d(self.px, self.p3, self.nproj * self.nb * 8, self.driver)

    def inner(self) -> None:
        self.stage("residual_to_t", self.residual_to_t)
        self.stage("project_t", self.project_t)
        self.stage("projected_gram", self.projected)
        self.stage("dsygvd", self.rr_solve)
        self.stage("rotation", self.rotate)

    def run_two_inner(self, x: np.ndarray, p: np.ndarray, eps: np.ndarray) -> dict:
        self.h2d(self.x, x)
        self.h2d(self.px, p)
        self.h2d(self.eps, eps)
        self.timing = {}
        self.inner()
        self.inner()
        xout, pout, eout = np.empty_like(x), np.empty_like(p), np.empty_like(eps)
        self.d2h(xout, self.x)
        self.d2h(pout, self.px)
        self.d2h(eout, self.eps)
        runtime.check(self.driver.cuCtxSynchronize(), "Davidson output")
        return {"x": xout, "p": pout, "eps": eout, "timing_s": self.timing}

    def close(self) -> None:
        if getattr(self, "solver", runtime.P()).value:
            SOLVER.cusolverDnDestroy(self.solver)
            self.solver = runtime.P()
        if getattr(self, "e4module", runtime.P()).value:
            self.driver.cuModuleUnload(self.e4module)
            self.e4module = runtime.P()
        self.module, self._blob = self.hreg_module, self.hreg_blob
        super().close()
