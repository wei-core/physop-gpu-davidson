# System requirements

The GPU backend requires an NVIDIA driver exposing the CUDA Driver API and a
CUDA toolkit containing NVRTC, cuBLAS, cuFFT and cuSOLVER. The code resolves
these libraries through the normal system loader; it does not assume a CUDA
installation directory or a particular GPU architecture.

Install GPAW PAW setup data separately and export `GPAW_SETUP_PATH`, or run
`gpaw install-data <directory>` before the first calculation. The reference
environment used Python 3.14.6, GPAW 25.7.0, ASE 3.29.0, NumPy 2.5.1, SciPy
1.18.0 and setup data 1.0.1.
