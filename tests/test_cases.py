from physop_gpu.materials.sic import CASES, case_config


def test_all_five_cases_are_geometry_policies():
    assert tuple(CASES) == ("SIC-008", "SIC-032", "SIC-064", "SIC-128", "SIC-216")
    assert case_config("SIC-216")["replication"] == (3, 3, 3)
    assert case_config("SIC-216")["nbands"] == 519
