# Reproducibility

Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`GPAW_NUM_THREADS=1` for the same deterministic threading boundary used by the
source validation. Run `python scripts/doctor.py`, then `python
scripts/smoke.py`. Source-machine timing values in `reference/` are historical
measurements, not portability or speedup gates.
