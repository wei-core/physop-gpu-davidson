# Execution model

GPAW owns outer SCF density, potential, occupations and mixing. At the live
seam it performs native `subspace_diagonalize`, the adapter refreshes `vt`,
`dH` and `dO`, the resident GPU solver runs two inner Davidson iterations, and
the updated `X/P/epsilon` is written back before GPAW continues.

The package modules correspond to E3 descriptor/H-S execution, E4 generic
Davidson and E5 live-SCF integration. `scripts/smoke.py` exercises only the
small SIC-008 case.
