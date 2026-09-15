from physop_gpu.davidson.contract import DavidsonShape, validate_graph


def test_production_davidson_contract():
    shape = DavidsonShape(nb=20, ng=710, nproj=104, batch=20)
    graph = validate_graph(shape)
    assert graph["m"] == 40
    assert graph["inner_iterations"] == 2
    assert graph["rr_backend"] == "cusolverDnDsygvd"
    assert graph["stale_HX_SX"] is False
