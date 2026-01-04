import pytest
import torch


def _available_devices():
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    return torch.device(request.param)
