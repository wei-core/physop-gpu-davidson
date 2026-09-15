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
    "${seed_python}" -m venv --system-site-packages "${repo_root}/.venv"
fi
python_bin="${repo_root}/.venv/bin/python"
if ! "${python_bin}" -c 'import ase, gpaw, numpy, scipy' >/dev/null 2>&1; then
    "${python_bin}" -m pip install -r "${repo_root}/environment/requirements.txt"
fi
site_packages="$("${python_bin}" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${repo_root}/src" > "${site_packages}/physop_gpu.pth"

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
