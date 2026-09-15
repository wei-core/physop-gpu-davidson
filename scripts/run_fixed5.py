#!/usr/bin/env python3
"""Convenience wrapper for the fixed-five-iteration CLI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physop_gpu.scf.live import run_case


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "SIC-008"
    print(run_case(case, "fixed5", "gpu"))
