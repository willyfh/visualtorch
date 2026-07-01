"""Tests for graph view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from visualtorch import graph_view


@pytest.fixture()
def dense_model() -> nn.Module:
    """A simple dense model creation."""

    class SimpleDense(nn.Module):
        """Simple Dense Model."""

        def __init__(self) -> None:
            super().__init__()
            self.h0 = nn.Linear(4, 8)
            self.h1 = nn.Linear(8, 8)
            self.h2 = nn.Linear(8, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Define the forward pass."""
            x = self.h0(x)
            x = self.h1(x)
            x = self.h2(x)
            return self.out(x)

    return SimpleDense()


@pytest.fixture()
def conv_model() -> nn.Module:
    """A simple conv model, exercising the Conv2d/ConvolutionBackward0 path."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.Conv2d(8, 16, 3, 1, 1),
    )


@pytest.fixture()
def wide_dense_model() -> nn.Module:
    """A dense model with a hidden layer wider than the default ellipsize_after threshold."""

    class WideDense(nn.Module):
        """A dense model with more than 10 hidden units, to trigger ellipsis drawing."""

        def __init__(self) -> None:
            super().__init__()
            self.h0 = nn.Linear(4, 20)
            self.out = nn.Linear(20, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.out(self.h0(x))

    return WideDense()


def test_dense_model_graph_view_runs(dense_model: nn.Module) -> None:
    """Test graph view using dense model."""
    _ = graph_view(dense_model, input_shape=(1, 4))


def test_conv_model_graph_view_runs(conv_model: nn.Module) -> None:
    """graph_view should support Conv2d layers, not just Linear."""
    img = graph_view(conv_model, input_shape=(1, 3, 16, 16))
    assert img is not None


def test_graph_view_ellipsizes_wide_layers(wide_dense_model: nn.Module) -> None:
    """A hidden layer wider than ellipsize_after should draw an ellipsis, not crash."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), ellipsize_after=10)
    assert img is not None


def test_graph_view_show_neurons_false(wide_dense_model: nn.Module) -> None:
    """show_neurons=False should render one node per layer instead of per neuron."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), show_neurons=False)
    assert img is not None


def test_graph_view_writes_to_file(dense_model: nn.Module, tmp_path: Path) -> None:
    """to_file should save a readable image to disk."""
    out_file = tmp_path / "graph.png"
    graph_view(dense_model, input_shape=(1, 4), to_file=str(out_file))

    assert out_file.exists()
    with Image.open(out_file) as saved_img:
        assert saved_img.size[0] > 0
        assert saved_img.size[1] > 0
