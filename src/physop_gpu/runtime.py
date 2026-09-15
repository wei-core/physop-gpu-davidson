"""Small CUDA Driver/NVRTC bridge used by the portable backend.

The original research scripts loaded WSL and CUDA-version-specific absolute
paths and compiled every kernel for one fixed architecture. This module keeps the same C
API, but resolves libraries through the system loader and asks the active
CUDA device for its compute capability before compiling a kernel.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import os
from dataclasses import dataclass

import numpy as np

P = C.c_void_p
U64 = C.c_uint64
I = C.c_int
U = C.c_uint
SZ = C.c_size_t
F = C.c_float


class CudaLibraryError(RuntimeError):
    """Raised when a required CUDA shared library cannot be resolved."""


def _library_candidates(name: str) -> list[str]:
    """Return loader names without assuming a CUDA installation directory."""
    aliases = {
        "cuda": ("cuda", "libcuda.so.1", "libcuda.so"),
        "nvrtc": ("nvrtc", "libnvrtc.so.12", "libnvrtc.so"),
        "cublas": ("cublas", "libcublas.so.12", "libcublas.so"),
        "cufft": ("cufft", "libcufft.so.11", "libcufft.so"),
        "cusolver": ("cusolver", "libcusolver.so.11", "libcusolver.so"),
    }
    names = list(aliases.get(name, (name,)))
    extra = os.environ.get("PHYSOP_CUDA_LIBRARY_PATH")
    if extra:
        names.extend(os.path.join(directory, candidate)
                     for directory in extra.split(os.pathsep) if directory
                     for candidate in aliases.get(name, (name,)))
    return names


def load_library(name: str) -> C.CDLL:
    """Load *name* with ``find_library``/normal dynamic-loader semantics."""
    errors: list[str] = []
    candidates = _library_candidates(name)
    discovered = ctypes.util.find_library(name)
    if discovered:
        candidates.insert(0, discovered)
    for candidate in dict.fromkeys(candidates):
        try:
            return C.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise CudaLibraryError(
        f"Unable to load CUDA component '{name}'. Install the CUDA driver/toolkit "
        f"component and make it visible to the dynamic loader (or set "
        f"PHYSOP_CUDA_LIBRARY_PATH). Tried: {', '.join(errors)}")


def _configure_nvrtc(lib: C.CDLL) -> C.CDLL:
    for name, args in {
        "nvrtcCreateProgram": [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P],
        "nvrtcCompileProgram": [P, I, P],
        "nvrtcGetCUBINSize": [P, C.POINTER(SZ)],
        "nvrtcGetCUBIN": [P, P],
        "nvrtcGetProgramLogSize": [P, C.POINTER(SZ)],
        "nvrtcGetProgramLog": [P, P],
        "nvrtcDestroyProgram": [C.POINTER(P)],
    }.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = I
    return lib


def _configure_driver(lib: C.CDLL) -> C.CDLL:
    for name, args in {
        "cuInit": [U],
        "cuDeviceGet": [C.POINTER(I), I],
        "cuDeviceGetName": [C.c_char_p, I, I],
        "cuDeviceComputeCapability": [C.POINTER(I), C.POINTER(I), I],
        "cuDeviceTotalMem_v2": [C.POINTER(SZ), I],
        "cuDriverGetVersion": [C.POINTER(I)],
        "cuCtxCreate_v2": [C.POINTER(P), U, I],
        "cuCtxDestroy_v2": [P],
        "cuCtxSynchronize": [],
        "cuModuleLoadData": [C.POINTER(P), P],
        "cuModuleUnload": [P],
        "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
        "cuMemAlloc_v2": [C.POINTER(U64), SZ],
        "cuMemFree_v2": [U64],
        "cuMemcpyHtoD_v2": [U64, P, SZ],
        "cuMemcpyDtoH_v2": [P, U64, SZ],
        "cuMemcpyDtoD_v2": [U64, U64, SZ],
        "cuEventCreate": [C.POINTER(P), U],
        "cuEventDestroy_v2": [P],
        "cuEventRecord": [P, P],
        "cuEventSynchronize": [P],
        "cuEventElapsedTime": [C.POINTER(F), P, P],
        "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, P, P],
    }.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = I
    return lib


def _configure_cufft(lib: C.CDLL) -> C.CDLL:
    for name in ("cufftPlanMany", "cufftExecD2Z", "cufftExecZ2D", "cufftDestroy"):
        getattr(lib, name).restype = I
    lib.cufftPlanMany.argtypes = [C.POINTER(I), I, C.POINTER(I), C.POINTER(I), I, I,
                                  C.POINTER(I), I, I, I, I]
    lib.cufftExecD2Z.argtypes = [I, P, P]
    lib.cufftExecZ2D.argtypes = [I, P, P]
    lib.cufftDestroy.argtypes = [I]
    return lib


def _configure_cublas(lib: C.CDLL) -> C.CDLL:
    for name, args in {
        "cublasCreate_v2": [C.POINTER(P)],
        "cublasDestroy_v2": [P],
        "cublasDgemm_v2": [P, I, I, I, I, I, P, P, I, P, I, P, P, I],
        "cublasDaxpy_v2": [P, I, P, P, I, P, I],
        "cublasDcopy_v2": [P, I, P, I, P, I],
    }.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = I
    return lib


def _configure_cusolver(lib: C.CDLL) -> C.CDLL:
    lib.cusolverDnCreate.argtypes = [C.POINTER(P)]
    lib.cusolverDnCreate.restype = I
    lib.cusolverDnDestroy.argtypes = [P]
    lib.cusolverDnDestroy.restype = I
    lib.cusolverDnDsygvd_bufferSize.argtypes = [P, I, I, I, I, P, I, P, I, P,
                                                C.POINTER(I)]
    lib.cusolverDnDsygvd_bufferSize.restype = I
    lib.cusolverDnDsygvd.argtypes = [P, I, I, I, I, P, I, P, I, P, P, I, P]
    lib.cusolverDnDsygvd.restype = I
    return lib


def _load_configured(name: str) -> C.CDLL:
    lib = load_library(name)
    return {"cuda": _configure_driver, "nvrtc": _configure_nvrtc,
            "cufft": _configure_cufft, "cublas": _configure_cublas,
            "cusolver": _configure_cusolver}[name](lib)


def check(code: int, where: str) -> None:
    if code:
        raise RuntimeError(f"{where}: CUDA/NVRTC status {code}")


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    major: int
    minor: int
    memory_bytes: int
    driver_version: int

    @property
    def compute_capability(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def nvrtc_architecture(self) -> bytes:
        return f"--gpu-architecture=sm_{self.major}{self.minor}".encode()


def libraries() -> dict[str, C.CDLL]:
    """Resolve all libraries required by the full GPU execution chain."""
    return {name: _load_configured(name)
            for name in ("cuda", "nvrtc", "cublas", "cufft", "cusolver")}


def device_info(driver: C.CDLL | None = None, index: int = 0) -> DeviceInfo:
    driver = driver or _load_configured("cuda")
    check(driver.cuInit(0), "cuInit")
    device = I()
    check(driver.cuDeviceGet(C.byref(device), index), "cuDeviceGet")
    name = C.create_string_buffer(256)
    check(driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
    major, minor = I(), I()
    check(driver.cuDeviceComputeCapability(C.byref(major), C.byref(minor), device),
          "cuDeviceComputeCapability")
    memory = SZ()
    check(driver.cuDeviceTotalMem_v2(C.byref(memory), device), "cuDeviceTotalMem")
    version = I()
    check(driver.cuDriverGetVersion(C.byref(version)), "cuDriverGetVersion")
    return DeviceInfo(index, name.value.decode(errors="replace"), major.value, minor.value,
                      memory.value, version.value)


def compile_cubin(source: str, name: bytes, nvrtc: C.CDLL, driver: C.CDLL) -> bytes:
    """Compile a kernel for the active device's runtime compute capability."""
    info = device_info(driver)
    program = P()
    check(nvrtc.nvrtcCreateProgram(C.byref(program), source.encode(), name, 0, None, None),
          "NVRTC create")
    try:
        options = (C.c_char_p * 1)(info.nvrtc_architecture)
        rc = nvrtc.nvrtcCompileProgram(program, 1, options)
        if rc:
            size = SZ()
            nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
            log = C.create_string_buffer(max(1, size.value))
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.value.decode(errors="replace"))
        size = SZ()
        check(nvrtc.nvrtcGetCUBINSize(program, C.byref(size)), "NVRTC CUBIN size")
        blob = C.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetCUBIN(program, blob), "NVRTC CUBIN")
        return blob.raw
    finally:
        nvrtc.nvrtcDestroyProgram(C.byref(program))


class GPU:
    """Driver context and common device-memory helpers."""

    def __init__(self) -> None:
        self.nvrtc = _load_configured("nvrtc")
        self.driver = _load_configured("cuda")
        check(self.driver.cuInit(0), "CUDA init")
        device = I()
        check(self.driver.cuDeviceGet(C.byref(device), 0), "CUDA device")
        self.device_info = device_info(self.driver)
        self.ctx = P()
        check(self.driver.cuCtxCreate_v2(C.byref(self.ctx), 0, device), "CUDA context")
        self.module = P()
        self._blob = None

    def load_module(self, source: str, name: bytes) -> None:
        self._blob = C.create_string_buffer(compile_cubin(source, name, self.nvrtc, self.driver))
        check(self.driver.cuModuleLoadData(C.byref(self.module), self._blob), "CUDA module")

    def close(self) -> None:
        if getattr(self, "module", P()).value:
            self.driver.cuModuleUnload(self.module)
            self.module = P()
        if getattr(self, "ctx", P()).value:
            self.driver.cuCtxDestroy_v2(self.ctx)
            self.ctx = P()

    def fn(self, name: str) -> P:
        function = P()
        check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()),
              f"CUDA function {name}")
        return function

    def alloc(self, nbytes: int) -> U64:
        pointer = U64()
        check(self.driver.cuMemAlloc_v2(C.byref(pointer), max(1, nbytes)), "CUDA alloc")
        return pointer

    def h2d(self, device: U64, host: np.ndarray) -> None:
        check(self.driver.cuMemcpyHtoD_v2(device, P(host.ctypes.data), host.nbytes), "H2D")

    def d2h(self, host: np.ndarray, device: U64) -> None:
        check(self.driver.cuMemcpyDtoH_v2(P(host.ctypes.data), device, host.nbytes), "D2H")

    def launch(self, function: P, grid: tuple[int, int], params: list) -> None:
        values, args = [], []
        for value in params:
            converted = U64(value.value) if hasattr(value, "value") else I(value)
            values.append(converted)
            args.append(C.cast(C.byref(converted), P))
        argv = (P * len(args))(*args)
        check(self.driver.cuLaunchKernel(function, grid[0], grid[1], 1, 256, 1, 1,
                                         0, None, argv, None), "CUDA launch")
