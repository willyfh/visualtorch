"""Tests for flow view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as func
from PIL import Image
from torch import nn
from visualtorch.flow import flow_view, layered_view


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


def test_sequential_model_flow_view_runs(sequential_model: nn.Sequential) -> None:
    """Test flow view on sequential model."""
    _ = flow_view(sequential_model, input_shape=(1, 3, 224, 224))


def test_module_list_model_flow_view_runs(module_list_model: nn.ModuleList) -> None:
    """Test flow view on module list model."""
    _ = flow_view(module_list_model, input_shape=(1, 3, 224, 224))


def test_custom_model_flow_view_runs(custom_model: nn.Module) -> None:
    """Test flow view on custom model."""
    _ = flow_view(custom_model, input_shape=(1, 3, 224, 224))


def test_lstm_model_flow_view_runs(lstm_model: nn.Module) -> None:
    """Test flow view on lstm model."""
    _ = flow_view(lstm_model, input_shape=(1, 10, 10))


def test_gru_model_flow_view_runs(gru_model: nn.Module) -> None:
    """Test flow view on gru model."""
    _ = flow_view(gru_model, input_shape=(1, 10, 10))


def test_rnn_model_flow_view_runs(rnn_model: nn.Module) -> None:
    """Test flow view on plain rnn model."""
    _ = flow_view(rnn_model, input_shape=(1, 10, 10))


@pytest.mark.parametrize("orientation", ["x", "y", "z"])
def test_flow_view_low_dim_orientation(classifier_model: nn.Module, orientation: str) -> None:
    """Test flow view on a model with a 1D output, for every supported orientation."""
    img = flow_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation=orientation)
    assert img is not None


def test_flow_view_invalid_low_dim_orientation_raises(classifier_model: nn.Module) -> None:
    """An unsupported low_dim_orientation should raise a clear ValueError."""
    with pytest.raises(ValueError, match="unsupported orientation"):
        flow_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation="bad")


def test_flow_view_one_dim_orientation_still_works(classifier_model: nn.Module) -> None:
    """The deprecated one_dim_orientation kwarg should still work and warn, not crash."""
    with pytest.warns(DeprecationWarning, match="one_dim_orientation"):
        deprecated_img = flow_view(classifier_model, input_shape=(1, 3, 16, 16), one_dim_orientation="x")
    current_img = flow_view(classifier_model, input_shape=(1, 3, 16, 16), low_dim_orientation="x")
    assert deprecated_img.tobytes() == current_img.tobytes()


def test_flow_view_with_type_ignore(sequential_model: nn.Sequential) -> None:
    """Layers matched by type_ignore should be skipped without error."""
    img = flow_view(
        sequential_model,
        input_shape=(1, 3, 224, 224),
        type_ignore=[nn.ReLU],
    )
    assert img is not None


def test_flow_view_with_legend(sequential_model: nn.Sequential) -> None:
    """legend=True should append a legend without error."""
    img = flow_view(sequential_model, input_shape=(1, 3, 224, 224), legend=True)
    assert img is not None


def test_flow_view_writes_to_file(sequential_model: nn.Sequential, tmp_path: Path) -> None:
    """to_file should save a readable image to disk."""
    out_file = tmp_path / "flow.png"
    flow_view(sequential_model, input_shape=(1, 3, 224, 224), to_file=str(out_file))

    assert out_file.exists()
    with Image.open(out_file) as saved_img:
        assert saved_img.size[0] > 0
        assert saved_img.size[1] > 0


def test_flow_view_with_show_dimension(sequential_model: nn.Sequential) -> None:
    """show_dimension=True should print each layer's shape without clipping or crashing."""
    img = flow_view(sequential_model, input_shape=(1, 3, 224, 224), show_dimension=True)
    assert img is not None


def test_flow_view_show_dimension_with_legend(sequential_model: nn.Sequential) -> None:
    """show_dimension and legend should be combinable."""
    img = flow_view(sequential_model, input_shape=(1, 3, 224, 224), show_dimension=True, legend=True)
    assert img is not None


def test_flow_view_output_size_matches_pre_refactor_baseline(sequential_model: nn.Sequential) -> None:
    """Locks in flow_view's canvas size across the backend/_volumetric_layout rewrite.

    Sizes captured from `main` (0b349a3) before `register_hook` was replaced with the shared
    `extract_architecture`/`layout_columns` backend - confirmed via a `git worktree` comparison
    at the time of the rewrite (see the graph_view equivalent test for why size, not an exact
    pixel hash: aggdraw's anti-aliasing isn't portable across platforms, but layout math is).

    Updated after `show_input` defaulted to True (an intentional, deliberate visual change - a
    single-consumer input box is now shown by default, unlike flow_view's original look), so
    these sizes are no longer literally pre-refactor but reflect the new intended default.

    "legend" was updated again after the synthetic input class was renamed from `InputDummyLayer`
    to `Input`, changing that legend patch's text width by 1px.
    """
    cases = {
        "default": flow_view(sequential_model, input_shape=(1, 3, 32, 32)),
        "no_volume": flow_view(sequential_model, input_shape=(1, 3, 32, 32), draw_volume=False),
        "show_dimension": flow_view(sequential_model, input_shape=(1, 3, 32, 32), show_dimension=True),
        "legend": flow_view(sequential_model, input_shape=(1, 3, 32, 32), legend=True),
        "type_ignore": flow_view(sequential_model, input_shape=(1, 3, 32, 32), type_ignore=[nn.ReLU]),
        "no_funnel": flow_view(sequential_model, input_shape=(1, 3, 32, 32), draw_funnel=False),
    }
    expected_sizes = {
        "default": (144, 42),
        "no_volume": (136, 32),
        "show_dimension": (353, 59),
        "legend": (148, 136),
        "type_ignore": (102, 42),
        "no_funnel": (144, 42),
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


def test_flow_view_residual_model_runs(residual_model: nn.Module) -> None:
    """flow_view should not crash on a model with a skip connection from the raw input."""
    img = flow_view(residual_model, input_shape=(1, 8, 16, 16))
    assert img is not None


def test_flow_view_hidden_skip_model_runs(hidden_skip_model: nn.Module) -> None:
    """flow_view should not crash on a model with a skip connection from a hidden layer."""
    img = flow_view(hidden_skip_model, input_shape=(1, 4))
    assert img is not None


def test_flow_view_residual_model_routes_above_diagram(residual_model: nn.Module) -> None:
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

    img_with_skip = flow_view(residual_model, input_shape=(1, 8, 16, 16))
    img_without_skip = flow_view(PlainChain(channels=8), input_shape=(1, 8, 16, 16))

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_flow_view_hidden_skip_model_routes_above_diagram(hidden_skip_model: nn.Module) -> None:
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

    img_with_skip = flow_view(hidden_skip_model, input_shape=(1, 4))
    img_without_skip = flow_view(PlainChain(), input_shape=(1, 4))

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_flow_view_deep_repeated_residual_blocks_stays_reasonably_sized() -> None:
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

    img_2_blocks = flow_view(DeepModel(4, 2), input_shape=(1, 4, 8, 8))
    img_6_blocks = flow_view(DeepModel(4, 6), input_shape=(1, 4, 8, 8))

    assert img_2_blocks.size[1] == img_6_blocks.size[1]


def _non_background_pixel_count(img: Image.Image) -> int:
    return int((np.array(img.convert("RGB")) != 255).any(axis=2).sum())


def test_flow_view_funnels_survive_large_de_differences_between_layers() -> None:
    """A funnel between two layers with very different 3D depth (`de`) must stay visible.

    Regression test for a real bug: drawing every connector first and every box second (instead
    of interleaving them column by column) let each box's opaque fill blot out large parts of
    its own incoming funnel whenever neighboring layers have a very different `de` - which
    barely showed on the small, near-constant-`de` models used to verify the flow_view
    rewrite, but was highly visible on a real CNN (found by the user manually comparing
    ReadTheDocs' `plot_basic_custom` example before/after the rewrite). Canvas *size* alone
    doesn't catch this class of bug (box positions are unaffected, only which pixels get
    painted), so this asserts on rendered content instead.
    """

    class SimpleCNN(nn.Module):
        """The exact model from docs/examples/flow/plot_basic_custom.py."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 28 * 28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with three shrinking conv/pool stages."""
            x = self.conv1(x)
            x = func.relu(x)
            x = func.max_pool2d(x, 2, 2)
            x = self.conv2(x)
            x = func.relu(x)
            x = func.max_pool2d(x, 2, 2)
            x = self.conv3(x)
            x = func.relu(x)
            x = func.max_pool2d(x, 2, 2)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = func.relu(x)
            return self.fc2(x)

    img = flow_view(SimpleCNN(), input_shape=(1, 3, 224, 224), legend=True)

    # Locked in from the current (fixed) implementation - confirmed pixel-identical to
    # pre-rewrite main (1ee630e) via a git-worktree comparison for this exact model. The buggy
    # intermediate version rendered thousands fewer non-background pixels here (missing funnel
    # segments), so a wide but real tolerance still catches a regression of that class.
    #
    # Updated after `show_input` defaulted to True - the input box is now shown, adding a
    # consistent amount of extra canvas/content on top of the original baseline above.
    assert img.size == (171, 364)
    non_bg = _non_background_pixel_count(img)
    error_msg = f"non-background pixel count {non_bg} outside expected range - funnel likely broken"
    assert 27000 <= non_bg <= 31000, error_msg


def test_flow_view_shows_all_input_boxes_for_multi_input_model() -> None:
    """Unlike the single-input case, flow_view must not hide any of 2+ separate input boxes -
    hiding one would make it ambiguous which arrow originates from which named input.
    """  # noqa: D205
    from visualtorch.backend import extract_architecture
    from visualtorch.utils.layer_utils import Input

    class TwoInputNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.head = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.head(torch.cat([self.a(x), self.b(y)], dim=1))

    architecture = extract_architecture(TwoInputNet(), ((1, 4), (1, 4)))
    input_labels = {layer.module.name() for layer in architecture.columns[0] if isinstance(layer.module, Input)}
    assert input_labels == {"input_0", "input_1"}

    img = flow_view(TwoInputNet(), input_shape=((1, 4), (1, 4)))
    assert img is not None


def test_flow_view_mismatched_depth_siamese_branches_needs_no_detour() -> None:
    """Sibling branches of different depths merging shouldn't trigger a routed detour."""

    class SiameseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.vector_branch = nn.Sequential(
                nn.Linear(10, 8),
                nn.ReLU(),
            )
            self.head = nn.Linear(16, 4)

        def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
            """Run each branch on its own input tensor, then concatenate and project."""
            image_features = self.image_branch(image)
            vector_features = self.vector_branch(vector)
            merged = torch.cat([image_features, vector_features], dim=1)
            return self.head(merged)

    class SiameseNetDepthMatched(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.vector_branch = nn.Sequential(
                nn.Linear(10, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
            )
            self.head = nn.Linear(16, 4)

        def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
            """Run each branch on its own input tensor, then concatenate and project."""
            image_features = self.image_branch(image)
            vector_features = self.vector_branch(vector)
            merged = torch.cat([image_features, vector_features], dim=1)
            return self.head(merged)

    input_shape = ((1, 3, 16, 16), (1, 10))
    img_mismatched = flow_view(SiameseNet(), input_shape=input_shape)
    img_matched = flow_view(SiameseNetDepthMatched(), input_shape=input_shape)

    assert img_mismatched.size[1] == img_matched.size[1]


def test_flow_view_low_dim_orientation_affects_2d_shapes() -> None:
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
        orientation: flow_view(model, input_shape=input_shape, low_dim_orientation=orientation).size
        for orientation in ("x", "y", "z")
    }

    assert len(set(sizes.values())) == 3, f"expected all 3 orientations to differ, got {sizes}"


def test_flow_view_2d_shape_seq_len_is_discarded() -> None:
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
    img_short_seq = flow_view(model, input_shape=(1, 5, 8))
    img_long_seq = flow_view(model, input_shape=(1, 50, 8))

    assert img_short_seq.tobytes() == img_long_seq.tobytes()

    model_bigger_hidden = SequenceClassifier(hidden_size=256)
    img_bigger_hidden = flow_view(model_bigger_hidden, input_shape=(1, 5, 8))

    assert img_short_seq.tobytes() != img_bigger_hidden.tobytes()


def test_layered_view_still_works(sequential_model: nn.Sequential) -> None:
    """The deprecated layered_view should still work, warn, and match flow_view's output."""
    with pytest.warns(DeprecationWarning, match="layered_view"):
        deprecated_img = layered_view(sequential_model, input_shape=(1, 3, 224, 224))
    current_img = flow_view(sequential_model, input_shape=(1, 3, 224, 224))
    assert deprecated_img.tobytes() == current_img.tobytes()


def test_layered_view_drops_index_ignore(sequential_model: nn.Sequential) -> None:
    """The removed index_ignore kwarg should be silently accepted, not raise."""
    with pytest.warns(DeprecationWarning, match="layered_view"):
        img = layered_view(sequential_model, input_shape=(1, 3, 224, 224), index_ignore=[0])
    assert img is not None
