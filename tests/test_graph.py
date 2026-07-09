"""Tests for graph view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from visualtorch.backend import extract_architecture
from visualtorch.graph import graph_view


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


@pytest.fixture()
def residual_model() -> nn.Module:
    """A model with a skip connection around a hidden sub-block, to prove branching is captured."""

    class ResidualBlock(nn.Module):
        """fc1->fc2 with a skip connection from stem's output, merged into out."""

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


@pytest.fixture()
def batchnorm_model() -> nn.Module:
    """A model using BatchNorm2d, a layer type never supported by the old hardcoded TARGET_OPS."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    )


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


def test_graph_view_residual_model_runs(residual_model: nn.Module) -> None:
    """graph_view should not crash on a model with a skip connection."""
    img = graph_view(residual_model, input_shape=(1, 4))
    assert img is not None


def test_graph_view_batchnorm_model_runs(batchnorm_model: nn.Module) -> None:
    """graph_view should not crash on a model using an untracked layer type (BatchNorm2d)."""
    img = graph_view(batchnorm_model, input_shape=(2, 3, 8, 8))
    assert img is not None


def test_extract_architecture_edge_count_dense_model(dense_model: nn.Module) -> None:
    """extract_architecture should produce exactly the expected edges for a known simple chain."""
    architecture = extract_architecture(dense_model, input_shape=(1, 4))

    # 4 chained Linear layers => 3 edges between them, plus 1 from the input dummy to the first
    # layer, and one layer per column (plus the synthetic input column).
    assert int(architecture.adjacency.sum()) == 4
    assert len(architecture.columns) == 5
    for column in architecture.columns:
        assert len(column) == 1


def test_extract_architecture_captures_branching(residual_model: nn.Module) -> None:
    """The merge point of a skip connection should have 2 incoming edges, not 1."""
    architecture = extract_architecture(residual_model, input_shape=(1, 4))

    out_node_id = f"{id(residual_model.out)}#0"
    in_degree = architecture.adjacency[:, architecture.id_to_index[out_node_id]].sum()

    assert in_degree == 2, "expected the merge node to have 2 incoming edges from the skip connection"


def test_extract_architecture_supports_untracked_layer_type(batchnorm_model: nn.Module) -> None:
    """A BatchNorm2d instance should actually appear among the traced layers."""
    architecture = extract_architecture(batchnorm_model, input_shape=(2, 3, 8, 8))

    traced_types = {type(wrapper.module) for column in architecture.columns for wrapper in column}
    assert nn.BatchNorm2d in traced_types


def test_graph_view_supports_weight_sharing() -> None:
    """A leaf module invoked more than once in one forward pass should not crash."""

    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc(x)
            return self.fc(x)

    img = graph_view(SharedModel(), input_shape=(1, 4))
    assert img is not None


def test_graph_view_bare_leaf_module_as_model() -> None:
    """Passing a single leaf module (no children) directly as the model should not crash."""
    img = graph_view(nn.Linear(4, 2), input_shape=(1, 4))
    assert img is not None


def test_extract_architecture_supports_module_list_as_model() -> None:
    """nn.ModuleList has no forward() of its own, so tracing must chain each child call manually."""
    model = nn.ModuleList([nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)])
    architecture = extract_architecture(model, input_shape=(1, 4))

    traced_types = {type(wrapper.module) for column in architecture.columns for wrapper in column}
    assert {nn.Linear, nn.ReLU}.issubset(traced_types)


def test_extract_architecture_wires_deeper_direct_input_consumer() -> None:
    """A node that reads the raw input directly, in addition to a hidden predecessor, still gets its input edge."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.a(x)
            return self.b(x + a)

    model = M()
    architecture = extract_architecture(model, input_shape=(1, 4))

    input_node_id = architecture.columns[0][0].node_id
    input_index = architecture.id_to_index[input_node_id]
    b_node_id = next(node_id for node_id in architecture.id_to_index if node_id.startswith(f"{id(model.b)}#"))

    assert architecture.adjacency[input_index, architecture.id_to_index[b_node_id]] == 1


def test_extract_architecture_survives_subclass_escaping_op() -> None:
    """Lineage should survive a leaf module whose forward escapes the tensor subclass internally."""

    class NumpyRoundTrip(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            arr = x.detach().numpy()
            return torch.from_numpy(arr.copy())

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.roundtrip = NumpyRoundTrip()
            self.b = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.a(x)
            x = self.roundtrip(x)
            return self.b(x)

    model = M()
    architecture = extract_architecture(model, input_shape=(1, 4))

    # input->a, a->roundtrip, roundtrip->b.
    assert int(architecture.adjacency.sum()) == 3
    assert [type(wrapper.module).__name__ for column in architecture.columns for wrapper in column] == [
        "Input",
        "Linear",
        "NumpyRoundTrip",
        "Linear",
    ]


def test_graph_view_with_show_dimension(dense_model: nn.Module) -> None:
    """show_dimension=True should print each layer's shape without clipping or crashing."""
    img = graph_view(dense_model, input_shape=(1, 4), show_dimension=True)
    assert img is not None


def test_graph_view_show_dimension_with_ellipsized_layer(wide_dense_model: nn.Module) -> None:
    """A wide, ellipsized layer's label should still be placed correctly."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), show_dimension=True, ellipsize_after=10)
    assert img is not None


def test_graph_view_show_dimension_with_branching(residual_model: nn.Module) -> None:
    """A column with a skip-connection merge should still get a correctly placed label."""
    img = graph_view(residual_model, input_shape=(1, 4), show_dimension=True)
    assert img is not None


def test_graph_view_mismatched_depth_siamese_branches_needs_no_detour() -> None:
    """Sibling branches of different depths merging shouldn't trigger a routed detour.

    Compares against a depth-matched control (the shorter branch padded to the same depth, so
    both branches merge at the same column, i.e. span == 1 - a case already known/confirmed to
    need zero detour rows) and asserts the two images have the same height. This is more robust
    than a hardcoded pixel baseline (an arbitrary implementation-detail number) and more precise
    than "with vs without the whole skip connection" (this bug is about a false-positive skip
    classification, not about whether a skip exists at all).
    """

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
        """Same topology, but vector_branch padded with an extra Linear(8, 8)+ReLU pair so both
        branches reach the merge point at the same column - a control case known to need 0
        detour levels today (span == 1 at the merge).
        """  # noqa: D205

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
    img_mismatched = graph_view(SiameseNet(), input_shape=input_shape, show_neurons=False)
    img_matched = graph_view(SiameseNetDepthMatched(), input_shape=input_shape, show_neurons=False)

    assert img_mismatched.size[1] == img_matched.size[1]


def test_graph_view_residual_model_routes_above_diagram(residual_model: nn.Module) -> None:
    """The skip connection should reserve extra vertical space rather than drawing invisibly."""

    class PlainChain(nn.Module):
        """The same 4 layers as residual_model, but without the skip connection."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with no skip connection."""
            return self.out(self.fc2(self.fc1(self.stem(x))))

    img_with_skip = graph_view(residual_model, input_shape=(1, 4), show_neurons=False)
    img_without_skip = graph_view(PlainChain(), input_shape=(1, 4), show_neurons=False)

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_graph_view_deep_repeated_residual_blocks_stays_reasonably_sized() -> None:
    """Back-to-back, non-overlapping residual blocks should share one detour level, not stack per block."""

    class ResBlock(nn.Module):
        """A minimal residual block: two convs with a skip connection, ending in a ReLU.

        The trailing ReLU matters: it's a leaf-module call, so it resets the tensor's producer
        set to just itself. Without it, `identity = x` would keep carrying every earlier block's
        producer forward untouched (mathematically correct - the merged tensor really is a sum
        of both - but it makes each block's skip nest inside the previous one instead of being
        independent, which is a different scenario than this test is after).
        """

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
        """A stack of N sequential, non-overlapping residual blocks."""

        def __init__(self, channels: int, n_blocks: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through every block in sequence."""
            for block in self.blocks:
                x = block(x)
            return x

    img_2_blocks = graph_view(DeepModel(4, 2), input_shape=(1, 4, 8, 8), show_neurons=False)
    img_6_blocks = graph_view(DeepModel(4, 6), input_shape=(1, 4, 8, 8), show_neurons=False)

    # Height should stay the same regardless of block count (every block's skip shares one
    # detour level); only width should grow as more layers are added.
    assert img_2_blocks.size[1] == img_6_blocks.size[1]


def test_graph_view_residual_model_show_neurons_true_runs(residual_model: nn.Module) -> None:
    """A skip edge under show_neurons=True should collapse to one connector, not crash on a dense mesh."""
    img = graph_view(residual_model, input_shape=(1, 4), show_neurons=True)
    assert img is not None


def test_graph_view_output_size_matches_pre_refactor_baseline(
    dense_model: nn.Module,
    conv_model: nn.Module,
    residual_model: nn.Module,
) -> None:
    """Locks in graph_view's canvas size across the backend/connectors extraction refactor.

    Sizes captured from `main` (1ee630e) before `model_to_adj_matrix`/`add_input_dummy_layer`
    and `_compute_skip_levels`/`_draw_connector` were moved into `visualtorch.backend`/
    `visualtorch.connectors` - confirmed byte-identical (not just same-size) via a `git worktree`
    hash comparison on a single machine at the time of the extraction. A per-pixel hash isn't
    portable across CI runners though: an identical hardcoded hash failed in CI (Linux) despite
    matching on macOS, because `aggdraw`'s anti-aliasing differs slightly by platform even though
    the underlying layout math - which is what this refactor could actually have broken - is
    identical. Canvas size is a platform-independent proxy for that layout math.
    """
    cases = {
        "dense_default": graph_view(dense_model, input_shape=(1, 4)),
        "dense_show_neurons_false": graph_view(dense_model, input_shape=(1, 4), show_neurons=False),
        "dense_show_dimension": graph_view(dense_model, input_shape=(1, 4), show_dimension=True),
        "conv_default": graph_view(conv_model, input_shape=(1, 3, 16, 16)),
        "residual_default": graph_view(residual_model, input_shape=(1, 4)),
        "residual_show_neurons_false": graph_view(residual_model, input_shape=(1, 4), show_neurons=False),
    }
    expected_sizes = {
        "dense_default": (1270, 490),
        "dense_show_neurons_false": (1270, 170),
        "dense_show_dimension": (1270, 507),
        "conv_default": (670, 610),
        "residual_default": (1270, 300),
        "residual_show_neurons_false": (1270, 220),
    }

    for name, img in cases.items():
        assert img.size == expected_sizes[name], f"{name} canvas size changed"


def test_graph_view_with_type_ignore(conv_model: nn.Module) -> None:
    """Layers matched by type_ignore should be skipped without error."""
    img = graph_view(
        conv_model,
        input_shape=(1, 3, 16, 16),
        type_ignore=[nn.ReLU],
    )
    assert img is not None


def test_graph_view_type_ignore_reduces_diagram(batchnorm_model: nn.Module) -> None:
    """type_ignore should produce a visually different (smaller) diagram than the default."""
    img_default = graph_view(batchnorm_model, input_shape=(2, 3, 8, 8))
    img_ignored = graph_view(batchnorm_model, input_shape=(2, 3, 8, 8), type_ignore=[nn.ReLU, nn.BatchNorm2d])
    assert img_ignored.tobytes() != img_default.tobytes()


def test_graph_view_outline_width_accepted(conv_model: nn.Module) -> None:
    """outline_width should visually change the rendered output."""
    img_default = graph_view(conv_model, input_shape=(1, 3, 16, 16), show_neurons=False)
    img_thick = graph_view(conv_model, input_shape=(1, 3, 16, 16), show_neurons=False, outline_width=5)
    assert img_thick.tobytes() != img_default.tobytes()
