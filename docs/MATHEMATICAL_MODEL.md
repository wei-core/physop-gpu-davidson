# Mathematical model

The portable tree preserves the validated E3/E4 model. The regular action is

`H_reg c = D_k c + P F^{-1}(v_R * F^{-1}(E c))`,

with the GPAW packed Gamma Q-map, R2C/C2R normalization and G=0 half weight.
PAW terms are `H_nl = B† dH B` and `S = I + B† dO B`. The descriptor is read
from the live GPAW calculation, so NG, NR, Nb, Nproj and the FFT shape are not
case constants.

The Davidson graph keeps `m = 2 Nb`, performs exactly two inner iterations,
uses `cusolverDnDsygvd(itype=1, uplo=lower)`, and applies one RR rotation to
both `[X,T]` and `[P,PT]`.
