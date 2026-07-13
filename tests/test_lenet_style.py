"""Tests for lenet view."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as func
from PIL import Image
from torch import nn

# lenet_view is deprecated in favor of render() - importing the private implementation directly
# so this file's tests (which exercise the actual rendering logic, not the deprecation wrapper
# itself) don't spam a DeprecationWarning on every call.
from visualtorch.lenet_style import _lenet_view as lenet_view
from visualtorch.lenet_style import lenet_view as deprecated_lenet_view


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
def gru_model() -> nn.Module:
    """Define a simple GRU model for testing."""

    class GRUModel(nn.Module):
        """A simple GRU model."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            out, _ = self.gru(x)
            return out

    return GRUModel(input_size=10, hidden_size=20, num_layers=2)


@pytest.fixture()
def rnn_model() -> nn.Module:
    """Define a simple plain RNN model for testing."""

    class RNNModel(nn.Module):
        """A simple RNN model."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            out, _ = self.rnn(x)
            return out

    return RNNModel(input_size=10, hidden_size=20, num_layers=2)


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


def test_lenet_view_is_deprecated(sequential_model: nn.Sequential) -> None:
    """The deprecated public lenet_view should still work, warn, and match render()'s output."""
    with pytest.warns(DeprecationWarning, match="lenet_view"):
        deprecated_img = deprecated_lenet_view(sequential_model, input_shape=(1, 3, 224, 224))
    current_img = lenet_view(sequential_model, input_shape=(1, 3, 224, 224))
    assert deprecated_img.tobytes() == current_img.tobytes()


def test_sequential_model_lenet_view_runs(sequential_model: nn.Sequential) -> None:
    """Test lenet view on sequential model."""
    _ = lenet_view(sequential_model, input_shape=(1, 3, 224, 224))


def test_module_list_model_lenet_view_runs(module_list_model: nn.ModuleList) -> None:
    """Test lenet view on module list model."""
    _ = lenet_view(module_list_model, input_shape=(1, 3, 224, 224))


def test_custom_model_lenet_view_runs(custom_model: nn.Module) -> None:
    """Test lenet view on custom model."""
    _ = lenet_view(custom_model, input_shape=(1, 3, 224, 224))


def test_lstm_model_lenet_view_runs(lstm_model: nn.Module) -> None:
    """Test lenet view on lstm model."""
    _ = lenet_view(lstm_model, input_shape=(1, 10, 10))


def test_gru_model_lenet_view_runs(gru_model: nn.Module) -> None:
    """Test lenet view on gru model."""
    _ = lenet_view(gru_model, input_shape=(1, 10, 10))


def test_rnn_model_lenet_view_runs(rnn_model: nn.Module) -> None:
    """Test lenet view on plain rnn model."""
    _ = lenet_view(rnn_model, input_shape=(1, 10, 10))


@pytest.mark.parametrize("orientation", ["x", "y", "z"])
def test_lenet_view_low_dim_orientation(classifier_model: nn.Module, orientation: str) -> None:
    """Test lenet view on a model with a 1D output, for every supported orientation."""
    img = lenet_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation=orientation)
    assert img is not None


def test_lenet_view_invalid_low_dim_orientation_raises(classifier_model: nn.Module) -> None:
    """An unsupported low_dim_orientation should raise a clear ValueError."""
    with pytest.raises(ValueError, match="unsupported orientation"):
        lenet_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation="bad")


def test_lenet_view_one_dim_orientation_still_works(classifier_model: nn.Module) -> None:
    """The deprecated one_dim_orientation kwarg should still work and warn, not crash."""
    with pytest.warns(DeprecationWarning, match="one_dim_orientation"):
        deprecated_img = lenet_view(classifier_model, input_shape=(1, 3, 16, 16), one_dim_orientation="x")
    current_img = lenet_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation="x")
    assert deprecated_img.tobytes() == current_img.tobytes()


def test_lenet_view_with_type_ignore(sequential_model: nn.Sequential) -> None:
    """Layers matched by type_ignore should be skipped without error."""
    img = lenet_view(
        sequential_model,
        input_shape=(1, 3, 224, 224),
        type_ignore=[nn.ReLU],
    )
    assert img is not None


def test_lenet_view_show_dimension_can_be_disabled(sequential_model: nn.Sequential) -> None:
    """show_dimension=False should drop labels and the reserved label-row height."""
    with_labels = lenet_view(sequential_model, input_shape=(1, 3, 32, 32))
    without_labels = lenet_view(sequential_model, input_shape=(1, 3, 32, 32), show_dimension=False)

    assert without_labels.height < with_labels.height


def test_lenet_view_writes_to_file(sequential_model: nn.Sequential, tmp_path: Path) -> None:
    """to_file should save a readable image to disk."""
    out_file = tmp_path / "lenet.png"
    lenet_view(sequential_model, input_shape=(1, 3, 224, 224), to_file=str(out_file))

    assert out_file.exists()
    with Image.open(out_file) as saved_img:
        assert saved_img.size[0] > 0
        assert saved_img.size[1] > 0


@pytest.fixture()
def small_sequential_model() -> nn.Sequential:
    """A smaller conv stack, so canvas sizes stay small enough to hardcode as a regression lock."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


def test_lenet_view_output_size_matches_post_rewrite_baseline(small_sequential_model: nn.Sequential) -> None:
    """Locks in lenet_view's canvas size across the backend/_volumetric_layout rewrite.

    Unlike graph_view/flow_view, this isn't identical to pre-rewrite `main`: the original
    `_create_architecture` budgeted extra vertical headroom for the offset-copy stack via a
    hand-tuned, not tightly-derived formula (`height + de*offset_z + 2*offset_z`, plus a flat
    +100 pixels for the label row). The rewrite replaces that with a consistently-derived
    margin, which was confirmed (via a `git worktree` comparison) to render visually identical
    for a non-branching model, with only a ~1% total height difference (1142px vs 1152px for a
    5-layer conv stack) from the fudge-factor gap - not pixel/size-identical, but a documented,
    inspected, intentional trade-off rather than an accidental regression.
    """
    cases = {
        "default": lenet_view(small_sequential_model, input_shape=(1, 3, 16, 16)),
        "no_funnel": lenet_view(small_sequential_model, input_shape=(1, 3, 16, 16), draw_funnel=False),
        "type_ignore": lenet_view(small_sequential_model, input_shape=(1, 3, 16, 16), type_ignore=[nn.ReLU]),
        "small_offset": lenet_view(small_sequential_model, input_shape=(1, 3, 16, 16), offset_z=5),
    }
    expected_sizes = {
        "default": (830, 286),
        "no_funnel": (830, 286),
        "type_ignore": (538, 286),
        "small_offset": (495, 206),
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
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection straight from the raw input."""
            identity = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            return self.relu(out + identity)

    return ResidualBlock(channels=4)


def test_lenet_view_residual_model_runs(residual_model: nn.Module) -> None:
    """lenet_view should not crash on a model with a skip connection."""
    img = lenet_view(residual_model, input_shape=(1, 4, 8, 8))
    assert img is not None


def test_lenet_view_residual_model_routes_above_diagram(residual_model: nn.Module) -> None:
    """A skip connection should reserve extra vertical space rather than drawing invisibly."""

    class PlainChain(nn.Module):
        """The same layers as residual_model, but without the skip connection."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with no skip connection."""
            return self.relu(self.conv2(self.relu(self.conv1(x))))

    img_with_skip = lenet_view(residual_model, input_shape=(1, 4, 8, 8))
    img_without_skip = lenet_view(PlainChain(channels=4), input_shape=(1, 4, 8, 8))

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_lenet_view_funnels_survive_large_de_differences_between_layers() -> None:
    """A funnel between two layers with very different depth/spread must stay visible.

    Regression test for the same class of bug fixed in flow_view: drawing every connector
    first and every box second (instead of interleaving them column by column) let each box's
    opaque fill blot out large parts of its own incoming funnel whenever neighboring layers have
    a very different `de`/`offset_z` spread.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
    )

    img = lenet_view(model, input_shape=(1, 3, 128, 128))

    # Locked in from the current (fixed) implementation.
    assert img.size == (1348, 558)
    non_bg = int((np.array(img.convert("RGB")) != 255).any(axis=2).sum())
    error_msg = f"non-background pixel count {non_bg} outside expected range - funnel likely broken"
    assert 110000 <= non_bg <= 145000, error_msg


def test_lenet_view_low_dim_orientation_affects_2d_shapes() -> None:
    """A 2D shape (e.g. an RNN's (seq_len, hidden_size)) should now respond to
    low_dim_orientation too, not just genuine 1D shapes.
    """  # noqa: D205

    class SequenceClassifier(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return out

    model = SequenceClassifier(hidden_size=64)
    input_shape = (1, 5, 8)

    sizes = {
        orientation: lenet_view(model, input_shape=input_shape, low_dim_orientation=orientation).size
        for orientation in ("x", "y", "z")
    }

    assert len(set(sizes.values())) == 3, f"expected all 3 orientations to differ, got {sizes}"


def test_lenet_view_2d_shape_seq_len_is_discarded() -> None:
    """The positional-like dim (e.g. seq_len) of a 2D shape shouldn't affect box size -
    only the feature-like dim (e.g. hidden_size) should.
    """  # noqa: D205

    class SequenceClassifier(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return out

    model = SequenceClassifier(hidden_size=64)
    img_short_seq = lenet_view(model, input_shape=(1, 5, 8))
    img_long_seq = lenet_view(model, input_shape=(1, 50, 8))

    assert img_short_seq.tobytes() == img_long_seq.tobytes()

    model_bigger_hidden = SequenceClassifier(hidden_size=256)
    img_bigger_hidden = lenet_view(model_bigger_hidden, input_shape=(1, 5, 8))

    assert img_short_seq.tobytes() != img_bigger_hidden.tobytes()


def test_lenet_view_connector_fill_and_width_accepted(residual_model: nn.Module) -> None:
    """connector_fill and connector_width should visually change the rendered output."""
    img_custom = lenet_view(
        residual_model,
        input_shape=(1, 4, 8, 8),
        connector_fill="blue",
        connector_width=3,
    )
    img_default = lenet_view(residual_model, input_shape=(1, 4, 8, 8))
    assert img_custom.tobytes() != img_default.tobytes()


def test_lenet_view_connector_fill_none_uses_box_outline(residual_model: nn.Module) -> None:
    """connector_fill=None (the default) should produce the same result as not passing it at all."""
    img_default = lenet_view(residual_model, input_shape=(1, 4, 8, 8))
    img_explicit_none = lenet_view(residual_model, input_shape=(1, 4, 8, 8), connector_fill=None)
    assert img_default.tobytes() == img_explicit_none.tobytes()


def test_lenet_view_connector_style_applies_to_regular_funnels(sequential_model: nn.Sequential) -> None:
    """Explicit connector styling should affect adjacent-layer funnels, not only skip edges."""
    img_default = lenet_view(sequential_model, input_shape=(1, 3, 224, 224))
    img_custom = lenet_view(
        sequential_model,
        input_shape=(1, 3, 224, 224),
        connector_fill="blue",
        connector_width=3,
    )
    assert img_custom.tobytes() != img_default.tobytes()


def test_lenet_view_outline_width_accepted(sequential_model: nn.Sequential) -> None:
    """outline_width should visually change the rendered output."""
    img_default = lenet_view(sequential_model, input_shape=(1, 3, 224, 224))
    img_thick = lenet_view(sequential_model, input_shape=(1, 3, 224, 224), outline_width=5)
    assert img_thick.tobytes() != img_default.tobytes()
