#!/usr/bin/env python3
"""Unified fixed5/converged native or GPU case entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=("SIC-008", "SIC-032", "SIC-064", "SIC-128", "SIC-216"))
    parser.add_argument("--mode", choices=("fixed5", "converged"), default="fixed5")
    parser.add_argument("--backend", choices=("native", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from physop_gpu.scf.live import run_case
    result = run_case(args.case, args.mode, args.backend, args.output)
    print(json.dumps({"case": args.case, "mode": args.mode, "backend": args.backend,
                      "terminal": result["terminal"], "scf_iterations": result["scf_iterations"],
                      "shape": result["shape"], "end_to_end_s": result["end_to_end_s"]}, indent=2))
    return 0 if result["terminal"] != "ERROR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
