import torch
import pytest


@pytest.fixture(autouse=True)
def reproducible_small_tests():
    torch.set_num_threads(1)
    torch.manual_seed(123)
