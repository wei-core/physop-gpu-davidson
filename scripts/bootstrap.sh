#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seed_python="${PYTHON:-}"
if [[ -z "${seed_python}" ]]; then
    if command -v python >/dev/null 2>&1; then
        seed_python="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        seed_python="$(command -v python3)"
    else
        echo "No Python interpreter found; set PYTHON=/path/to/python" >&2
        exit 1
    fi
fi
if [[ ! -x "${repo_root}/.venv/bin/python" ]]; then
    "${seed_python}" -m venv "${repo_root}/.venv"
fi
python_bin="${repo_root}/.venv/bin/python"
"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -e "${repo_root}"

if [[ -n "${GPAW_SETUP_PATH:-}" ]]; then
    echo "Using GPAW_SETUP_PATH=${GPAW_SETUP_PATH}"
else
    echo "GPAW_SETUP_PATH is not set. Install GPAW PAW setups with:"
    echo "  gpaw install-data <directory>"
    echo "Then export GPAW_SETUP_PATH=<directory> before running cases."
fi
echo "Bootstrap complete. Activate the environment first:"
echo "  source ${repo_root}/.venv/bin/activate"
echo "Then run: python scripts/doctor.py"
