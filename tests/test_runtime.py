from physop_gpu.runtime import DeviceInfo


def test_nvrtc_architecture_comes_from_device_capability():
    info = DeviceInfo(0, "test", 9, 0, 1, 12000)
    assert info.nvrtc_architecture == b"--gpu-architecture=sm_90"
