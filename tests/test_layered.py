import pytest
import torch.nn as nn
from visualtorch import layered_view


@pytest.fixture
def sequential_model():
    # Define or load an example Sequential torch model for testing
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    return model


@pytest.fixture
def module_list_model():
    # Define or load an example ModuleList-based torch model for testing
    model = nn.ModuleList(
        [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]
    )
    return model


def test_sequential_model_layered_view_runs(sequential_model):
    try:
        _ = layered_view(sequential_model)
    except Exception as e:
        pytest.fail(f"layered_view raised an exception with Sequential model: {e}")


def test_module_list_model_layered_view_runs(module_list_model):
    try:
        _ = layered_view(module_list_model)
    except Exception as e:
        pytest.fail(f"layered_view raised an exception with ModuleList model: {e}")
