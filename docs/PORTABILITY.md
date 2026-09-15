# Portability

CUDA libraries are resolved with `ctypes.util.find_library` and ordinary
dynamic-loader names. `PHYSOP_CUDA_LIBRARY_PATH` is an optional search-path
override. NVRTC obtains `sm_<major><minor>` from
`cuDeviceComputeCapability`; no source GPU architecture is compiled into the
runtime.

The only GPAW dependency is the installed GPAW authority and its PAW setup
data. No historical V5R path is imported at runtime.
