"""Tests for layered view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
import torch.nn.functional as func
from PIL import Image
from torch import nn
from visualtorch import layered_view


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


@pytest.fixture()
def lstm_model() -> nn.Module:
    """Define a simple LSTM model for testing."""

    class LSTMModel(nn.Module):
        """A simple LSTM model."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            out, _ = self.lstm(x)
            return out

    # Create an instance of the LSTM model
    return LSTMModel(input_size=10, hidden_size=20, num_layers=2)


@pytest.fixture()
def classifier_model() -> nn.Module:
    """Define a model ending in a 1D (per-sample) output, e.g. classification logits."""

    class ClassifierModel(nn.Module):
        """A cnn model that ends with a 1D output."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = self.conv(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    return ClassifierModel()


def test_sequential_model_layered_view_runs(sequential_model: nn.Sequential) -> None:
    """Test layered view on sequential model."""
    _ = layered_view(sequential_model, input_shape=(1, 3, 224, 224))


def test_module_list_model_layered_view_runs(module_list_model: nn.ModuleList) -> None:
    """Test layered view on module list model."""
    _ = layered_view(module_list_model, input_shape=(1, 3, 224, 224))


def test_custom_model_layered_view_runs(custom_model: nn.Module) -> None:
    """Test layered view on custom model."""
    _ = layered_view(custom_model, input_shape=(1, 3, 224, 224))


def test_lstm_model_layered_view_runs(lstm_model: nn.Module) -> None:
    """Test layered view on lstm model."""
    _ = layered_view(lstm_model, input_shape=(1, 10, 10))


@pytest.mark.parametrize("orientation", ["x", "y", "z"])
def test_layered_view_one_dim_orientation(classifier_model: nn.Module, orientation: str) -> None:
    """Test layered view on a model with a 1D output, for every supported orientation."""
    img = layered_view(classifier_model, input_shape=(1, 3, 16, 16), one_dim_orientation=orientation)
    assert img is not None


def test_layered_view_invalid_one_dim_orientation_raises(classifier_model: nn.Module) -> None:
    """An unsupported one_dim_orientation should raise a clear ValueError."""
    with pytest.raises(ValueError, match="unsupported orientation"):
        layered_view(classifier_model, input_shape=(1, 3, 16, 16), one_dim_orientation="bad")


def test_layered_view_with_type_ignore(sequential_model: nn.Sequential) -> None:
    """Layers matched by type_ignore should be skipped without error."""
    img = layered_view(
        sequential_model,
        input_shape=(1, 3, 224, 224),
        type_ignore=[nn.ReLU],
    )
    assert img is not None


def test_layered_view_with_legend(sequential_model: nn.Sequential) -> None:
    """legend=True should append a legend without error."""
    img = layered_view(sequential_model, input_shape=(1, 3, 224, 224), legend=True)
    assert img is not None


def test_layered_view_writes_to_file(sequential_model: nn.Sequential, tmp_path: Path) -> None:
    """to_file should save a readable image to disk."""
    out_file = tmp_path / "layered.png"
    layered_view(sequential_model, input_shape=(1, 3, 224, 224), to_file=str(out_file))

    assert out_file.exists()
    with Image.open(out_file) as saved_img:
        assert saved_img.size[0] > 0
        assert saved_img.size[1] > 0


def test_layered_view_with_show_dimension(sequential_model: nn.Sequential) -> None:
    """show_dimension=True should print each layer's shape without clipping or crashing."""
    img = layered_view(sequential_model, input_shape=(1, 3, 224, 224), show_dimension=True)
    assert img is not None


def test_layered_view_show_dimension_with_legend(sequential_model: nn.Sequential) -> None:
    """show_dimension and legend should be combinable."""
    img = layered_view(sequential_model, input_shape=(1, 3, 224, 224), show_dimension=True, legend=True)
    assert img is not None


def test_layered_view_output_size_matches_pre_refactor_baseline(sequential_model: nn.Sequential) -> None:
    """Locks in layered_view's canvas size across the backend/_volumetric_layout rewrite.

    Sizes captured from `main` (0b349a3) before `register_hook` was replaced with the shared
    `extract_architecture`/`layout_columns` backend - confirmed via a `git worktree` comparison
    at the time of the rewrite (see the graph_view equivalent test for why size, not an exact
    pixel hash: aggdraw's anti-aliasing isn't portable across platforms, but layout math is).
    """
    cases = {
        "default": layered_view(sequential_model, input_shape=(1, 3, 32, 32)),
        "no_volume": layered_view(sequential_model, input_shape=(1, 3, 32, 32), draw_volume=False),
        "show_dimension": layered_view(sequential_model, input_shape=(1, 3, 32, 32), show_dimension=True),
        "legend": layered_view(sequential_model, input_shape=(1, 3, 32, 32), legend=True),
        "type_ignore": layered_view(sequential_model, input_shape=(1, 3, 32, 32), type_ignore=[nn.ReLU]),
        "no_funnel": layered_view(sequential_model, input_shape=(1, 3, 32, 32), draw_funnel=False),
    }
    expected_sizes = {
        "default": (124, 42),
        "no_volume": (116, 32),
        "show_dimension": (303, 59),
        "legend": (124, 136),
        "type_ignore": (82, 42),
        "no_funnel": (124, 42),
    }

    for name, img in cases.items():
        assert img.size == expected_sizes[name], f"{name} canvas size changed"


@pytest.fixture()
def residual_model() -> nn.Module:
    """A residual block whose shortcut is the model's own raw input (the most common pattern)."""

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection straight from the raw input."""
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + identity
            return self.relu(out)

    return ResidualBlock(channels=8)


@pytest.fixture()
def hidden_skip_model() -> nn.Module:
    """A residual block whose shortcut originates from a hidden layer, not the raw input."""

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection around fc1/fc2."""
            stem_out = self.stem(x)
            branch = self.fc2(self.fc1(stem_out))
            merged = branch + stem_out
            return self.out(merged)

    return ResidualBlock()


def test_layered_view_residual_model_runs(residual_model: nn.Module) -> None:
    """layered_view should not crash on a model with a skip connection from the raw input."""
    img = layered_view(residual_model, input_shape=(1, 8, 16, 16))
    assert img is not None


def test_layered_view_hidden_skip_model_runs(hidden_skip_model: nn.Module) -> None:
    """layered_view should not crash on a model with a skip connection from a hidden layer."""
    img = layered_view(hidden_skip_model, input_shape=(1, 4))
    assert img is not None


def test_layered_view_residual_model_routes_above_diagram(residual_model: nn.Module) -> None:
    """A skip connection from the raw input should reserve extra vertical space, not vanish."""

    class PlainChain(nn.Module):
        """The same layers as residual_model, but without the skip connection."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with no skip connection."""
            return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

    img_with_skip = layered_view(residual_model, input_shape=(1, 8, 16, 16))
    img_without_skip = layered_view(PlainChain(channels=8), input_shape=(1, 8, 16, 16))

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_layered_view_hidden_skip_model_routes_above_diagram(hidden_skip_model: nn.Module) -> None:
    """A skip connection from a hidden layer should also reserve extra vertical space."""

    class PlainChain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with no skip connection."""
            return self.out(self.fc2(self.fc1(self.stem(x))))

    img_with_skip = layered_view(hidden_skip_model, input_shape=(1, 4))
    img_without_skip = layered_view(PlainChain(), input_shape=(1, 4))

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_layered_view_deep_repeated_residual_blocks_stays_reasonably_sized() -> None:
    """Back-to-back, non-overlapping residual blocks should share one detour level, not stack per block."""

    class ResBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection around conv1/conv2."""
            identity = x
            out = self.conv2(self.conv1(x))
            return self.relu(out + identity)

    class DeepModel(nn.Module):
        def __init__(self, channels: int, n_blocks: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through every block in sequence."""
            for block in self.blocks:
                x = block(x)
            return x

    img_2_blocks = layered_view(DeepModel(4, 2), input_shape=(1, 4, 8, 8))
    img_6_blocks = layered_view(DeepModel(4, 6), input_shape=(1, 4, 8, 8))

    assert img_2_blocks.size[1] == img_6_blocks.size[1]
