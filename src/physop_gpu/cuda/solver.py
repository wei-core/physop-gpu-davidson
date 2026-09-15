"""cuSOLVER Dsygvd bindings used by the projected Davidson solve."""
from __future__ import annotations

from .. import runtime

SOLVER = runtime._load_configured("cusolver")
CUSOLVER_EIG_TYPE_1 = 1
CUSOLVER_EIG_MODE_VECTOR = 1
CUBLAS_FILL_MODE_LOWER = 0
