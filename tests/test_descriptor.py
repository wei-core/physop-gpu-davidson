import numpy as np

from physop_gpu.descriptor import (GeneralDescriptor, PawLayout,
                                   REPRESENTATION_GAMMA_PACKED, _gamma_completion)


def descriptor() -> GeneralDescriptor:
    shape = (2, 2, 2)
    q = np.arange(4, dtype=np.int64)
    return GeneralDescriptor(
        case="TEST", representation=REPRESENTATION_GAMMA_PACKED, n_atom=1,
        nb=2, ng=4, nr=8, fft_shape=shape, q_map=q,
        kinetic=np.ones(4), potential=np.ones(8),
        projector_integrate=np.ones((8, 3)), projector_add=np.ones((8, 3)),
        dH=np.eye(3), dO=np.eye(3), paw=PawLayout((3,), (0, 3)),
        gamma_self_conjugate=np.array([], dtype=np.int64),
        gamma_mirror_target=np.array([6], dtype=np.int64),
        gamma_mirror_source=np.array([1], dtype=np.int64))


def test_descriptor_validation_is_runtime_shape_driven():
    assert descriptor().validate()["status"] == "PASS"


def test_gamma_completion_has_valid_packed_indices():
    self_conjugate, targets, sources = _gamma_completion(np.arange(4), (2, 2, 2))
    assert self_conjugate.dtype == np.int64
    assert targets.dtype == np.int64
    assert sources.dtype == np.int64
    assert np.all(targets < 2 * 2 * (2 // 2 + 1))
