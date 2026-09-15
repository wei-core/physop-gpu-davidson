#!/usr/bin/env python3
"""Report Python/GPAW/CUDA readiness without assuming a source-machine path."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def package_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "MISSING"
    module = __import__(name)
    return str(getattr(module, "__version__", getattr(module, "version", "installed")))


def main() -> int:
    print(f"Python           = {sys.version.split()[0]} ({platform.platform()})")
    for name in ("gpaw", "ase", "numpy", "scipy"):
        try:
            print(f"{name.capitalize():16}= {package_version(name)}")
        except Exception as exc:
            print(f"{name.capitalize():16}= ERROR: {exc}")
    print(f"GPAW_SETUP_PATH  = {os.environ.get('GPAW_SETUP_PATH', 'unset')}")
    try:
        from physop_gpu import runtime
        libs = runtime.libraries()
        info = runtime.device_info(libs["cuda"])
        print(f"GPU              = {info.name}")
        print(f"Compute capability= {info.compute_capability}")
        print(f"GPU memory       = {info.memory_bytes / 1024**3:.2f} GiB")
        print(f"CUDA driver      = {info.driver_version}")
        for name in ("cuda", "nvrtc", "cublas", "cufft", "cusolver"):
            print(f"{name:16}= OK ({libs[name]._name})")
        if importlib.util.find_spec("gpaw") is None:
            raise RuntimeError("GPAW is not installed")
        print("PHYSOP_RUNTIME_READY")
        return 0
    except Exception as exc:
        print(f"PHYSOP_RUNTIME_NOT_READY: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
