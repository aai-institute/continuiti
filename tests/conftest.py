import pytest
import random
import torch

pytest_plugins = [
    "tests.transforms.fixtures",
]


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(0)
    torch.manual_seed(0)
