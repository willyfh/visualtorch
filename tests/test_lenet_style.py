"""Tests for lenet view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as func
from torch import nn
from visualtorch import lenet_view


@pytest.fixture()
def sequential_model() -> nn.Sequential:
    """Define Sequential torch model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


@pytest.fixture()
def module_list_model() -> nn.ModuleList:
    """Define ModuleList-based torch model for testing."""
    return nn.ModuleList(
        [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ],
    )


@pytest.fixture()
def custom_model() -> nn.Module:
    """Define the custom model."""

    class CustomModel(nn.Module):
        """A simple custom cnn model."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Funcorward pass."""
            x = func.relu(self.conv1(x))
            x = func.relu(self.conv2(x))
            return func.max_pool2d(x, 2, 2)

    # Create an instance of the custom model
    return CustomModel()


def test_sequential_model_lenet_view_runs(sequential_model: nn.Sequential) -> None:
    """Test lenet view on sequential model."""
    _ = lenet_view(sequential_model, input_shape=(1, 3, 224, 224))


def test_module_list_model_lenet_view_runs(module_list_model: nn.ModuleList) -> None:
    """Test lenet view on module list model."""
    _ = lenet_view(module_list_model, input_shape=(1, 3, 224, 224))


def test_custom_model_lenet_view_runs(custom_model: nn.Module) -> None:
    """Test lenet view on custom model."""
    _ = lenet_view(custom_model, input_shape=(1, 3, 224, 224))
