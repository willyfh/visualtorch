"""Regression tests for shape-handling crashes reported in open GitHub issues.

See https://github.com/willyfh/visualtorch/issues/63, /68, /69.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import pytest
import torch
from torch import nn
from visualtorch import layered_view, lenet_view
from visualtorch.utils.utils import self_multiply


@pytest.fixture()
def multi_output_container_model() -> nn.Module:
    """A model whose inner block is a container (not Sequential/ModuleList) that returns multiple tensors.

    This mirrors timm's ``FeatureListNet`` used in issue #69: a container module that isn't
    ``nn.Sequential``/``nn.ModuleList`` and returns multiple differently-shaped tensors.
    """

    class FeaturePyramidBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
            self.stage2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
            self.stage3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        def forward(self, x: torch.Tensor) -> list:
            f1 = self.stage1(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            return [f1, f2, f3]

    class MultiScaleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = FeaturePyramidBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.features(x)[-1]

    return MultiScaleNet()


def test_layered_view_multi_output_container(multi_output_container_model: nn.Module) -> None:
    """layered_view should not crash on container modules that output multiple tensors."""
    img = layered_view(multi_output_container_model, input_shape=(1, 3, 32, 32))
    assert img is not None


def test_lenet_view_multi_output_container(multi_output_container_model: nn.Module) -> None:
    """lenet_view should not crash on container modules that output multiple tensors."""
    img = lenet_view(multi_output_container_model, input_shape=(1, 3, 32, 32))
    assert img is not None


def test_self_multiply_handles_nested_shape() -> None:
    """self_multiply should always reduce to a scalar, even if an element is itself a shape."""
    nested_shape = (1, torch.Size([4, 8]))
    result = self_multiply(nested_shape)
    assert isinstance(result, int)


def test_lenet_view_rejects_multi_input_shape_with_clear_error() -> None:
    """Multi-tensor-input models aren't supported yet; the failure should be a clear ValueError."""
    model = nn.Linear(10, 5)
    multi_input_shape = ((1, 10), (1, 10))

    with pytest.raises(ValueError, match="single"):
        lenet_view(model, input_shape=multi_input_shape)
