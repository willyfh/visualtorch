import pytest
import torch.nn as nn
import torch.nn.functional as F

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


@pytest.fixture
def custom_model():
    # Define the custom model inside the function
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            return x

    # Create an instance of the custom model
    model = CustomModel()
    return model


def test_sequential_model_layered_view_runs(sequential_model):
    try:
        _ = layered_view(sequential_model, input_shape=(1, 3, 224, 224))
    except Exception as e:
        pytest.fail(f"layered_view raised an exception with Sequential model: {e}")


def test_module_list_model_layered_view_runs(module_list_model):
    try:
        _ = layered_view(module_list_model, input_shape=(1, 3, 224, 224))
    except Exception as e:
        pytest.fail(f"layered_view raised an exception with ModuleList model: {e}")


def test_custom_model_layered_view_runs(custom_model):
    try:
        _ = layered_view(custom_model, input_shape=(1, 3, 224, 224))
    except Exception as e:
        pytest.fail(f"layered_view raised an exception with Custom model: {e}")
