"""Tests for visualtorch.animate(), the unified entry point for animated rendering.

The underlying per-style implementations (`_graph_view_animate`/`_flow_view_animate`/
`_lenet_view_animate`) are intentionally private and never imported directly here - matching how
`test_render.py` only ever exercises `render()`, never the private `_render_graph`/etc. helpers.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest
import torch
from PIL import Image, ImageSequence
from torch import nn
from visualtorch import animate
from visualtorch.backend import extract_architecture

# graph_view/flow_view/lenet_view are deprecated in favor of render() - importing the private
# implementations directly (used here only to compare animate()'s last frame against the static
# render for byte-identical output) so this doesn't spam a DeprecationWarning on every call.
from visualtorch.flow import _flow_view as flow_view
from visualtorch.graph import _graph_view as graph_view
from visualtorch.lenet_style import _lenet_view as lenet_view

_STATIC_VIEW_FUNCS: dict[str, Callable] = {
    "graph": graph_view,
    "flow": flow_view,
    "lenet": lenet_view,
}
_STYLES = list(_STATIC_VIEW_FUNCS)


@pytest.fixture
def sequential_model() -> nn.Module:
    """A simple, single-input model with no branching."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1, 1),
    )


@pytest.fixture
def residual_model() -> nn.Module:
    """A model with a skip connection, to prove the merge column animates correctly."""

    class ResidualBlock(nn.Module):
        """conv1/bn1/relu/conv2/bn2 with a skip connection around them."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(4, 4, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection around conv1/bn1/relu/conv2/bn2."""
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + identity
            return self.relu(out)

    return ResidualBlock()


@pytest.fixture
def siamese_model() -> nn.Module:
    """A two-input model, to prove parallel inputs reveal together, in the same frame."""

    class SiameseNet(nn.Module):
        """An image branch and a tabular-vector branch, merged by concat."""

        def __init__(self) -> None:
            super().__init__()
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 8, 3, 1, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.vector_branch = nn.Sequential(nn.Linear(10, 8), nn.ReLU())
            self.head = nn.Linear(16, 4)

        def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
            """Run each branch on its own input, then concatenate and project."""
            merged = torch.cat([self.image_branch(image), self.vector_branch(vector)], dim=1)
            return self.head(merged)

    return SiameseNet()


@pytest.fixture
def inception_model() -> nn.Module:
    """A multi-branch (not just a residual skip) model: parallel paths merged by addition."""

    class InceptionBlock(nn.Module):
        """3 parallel conv branches on the same input, merged by addition."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.branch1 = nn.Conv2d(channels, channels, kernel_size=1)
            self.branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.branch3 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run every branch on the same input, then merge by addition."""
            return self.branch1(x) + self.branch2(x) + self.branch3(x)

    return InceptionBlock(channels=4)


def _non_background_pixel_count(img: Image.Image, background_fill: str = "white") -> int:
    """Count pixels differing from `background_fill`, as a proxy for "how much has been drawn"."""
    bg = Image.new("RGB", (1, 1), background_fill).getpixel((0, 0))
    return sum(1 for pixel in img.convert("RGB").getdata() if pixel != bg)


@pytest.mark.parametrize("style", _STYLES)
def test_frame_count_matches_column_count(style: str, sequential_model: nn.Module) -> None:
    """Animating a plain sequential model yields one frame per traced column."""
    input_shape = (1, 3, 8, 8)
    frames = animate(sequential_model, input_shape, style=style)
    architecture = extract_architecture(sequential_model, input_shape)
    assert len(frames) == len(architecture.columns)


@pytest.mark.parametrize("style", _STYLES)
def test_ink_is_non_decreasing_across_frames(style: str, residual_model: nn.Module) -> None:
    """Nothing disappears frame to frame - each frame reveals strictly more than the last."""
    frames = animate(residual_model, (1, 4, 8, 8), style=style)
    ink_counts = [_non_background_pixel_count(frame) for frame in frames]
    assert ink_counts == sorted(ink_counts)
    assert ink_counts[0] < ink_counts[-1]


@pytest.mark.parametrize("style", _STYLES)
@pytest.mark.parametrize("extra_kwargs", [{}, {"show_dimension": True}])
def test_last_frame_matches_static_render(style: str, residual_model: nn.Module, extra_kwargs: dict) -> None:
    """The fully-revealed last frame must be byte-identical to the equivalent static render."""
    input_shape = (1, 4, 8, 8)
    frames = animate(residual_model, input_shape, style=style, **extra_kwargs)
    static_img = _STATIC_VIEW_FUNCS[style](residual_model, input_shape, **extra_kwargs)
    assert frames[-1].tobytes() == static_img.tobytes()


def test_graph_last_frame_matches_static_show_neurons_true() -> None:
    """graph_view's show_neurons=True path (unique to graph style) also matches statically."""
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    input_shape = (1, 4)
    frames = animate(model, input_shape, style="graph", show_neurons=True)
    static_img = graph_view(model, input_shape, show_neurons=True)
    assert frames[-1].tobytes() == static_img.tobytes()


def test_flow_legend_fixed_and_correct_from_first_frame() -> None:
    """Flow's animated legend renders full/fixed from frame 0, matching the static legend."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, 1, 1), nn.BatchNorm2d(4), nn.ReLU())
    input_shape = (1, 3, 8, 8)

    no_legend_static = flow_view(model, input_shape, legend=False)
    static_with_legend = flow_view(model, input_shape, legend=True)
    frames = animate(model, input_shape, style="flow", legend=True)

    assert frames[-1].tobytes() == static_with_legend.tobytes()

    # legend_position defaults to "bottom-left": the legend strip is appended below the
    # diagram, so its height is exactly the difference between the legend-on and legend-off
    # canvas heights, regardless of any width padding differences between the two.
    legend_strip_height = static_with_legend.height - no_legend_static.height
    assert legend_strip_height > 0

    def legend_crop(img: Image.Image) -> Image.Image:
        return img.crop((0, img.height - legend_strip_height, img.width, img.height))

    # The legend is drawn full/fixed from frame 0 onward - even though the diagram above it is
    # still mostly unrevealed, the legend strip itself should already match the final frame.
    assert legend_crop(frames[0]).tobytes() == legend_crop(frames[-1]).tobytes()


@pytest.mark.parametrize("style", ["graph", "lenet"])
def test_graph_and_lenet_legends_are_fixed_from_first_frame(style: str) -> None:
    """Graph and LeNet animations should show the complete, fixed legend on every frame."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, 1, 1), nn.BatchNorm2d(4), nn.ReLU())
    input_shape = (1, 3, 8, 8)

    no_legend_static = _STATIC_VIEW_FUNCS[style](model, input_shape, legend=False)
    static_with_legend = _STATIC_VIEW_FUNCS[style](model, input_shape, legend=True)
    frames = animate(model, input_shape, style=style, legend=True)

    assert frames[-1].tobytes() == static_with_legend.tobytes()

    legend_strip_height = static_with_legend.height - no_legend_static.height
    assert legend_strip_height > 0

    def legend_crop(img: Image.Image) -> Image.Image:
        return img.crop((0, img.height - legend_strip_height, img.width, img.height))

    assert legend_crop(frames[0]).tobytes() == legend_crop(frames[-1]).tobytes()


@pytest.mark.parametrize("style", ["graph", "lenet"])
def test_graph_and_lenet_animate_forwards_legend_position(style: str) -> None:
    """Animated graph and LeNet renders should place legends like static renders."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, 1, 1), nn.BatchNorm2d(4), nn.ReLU())
    input_shape = (1, 3, 8, 8)

    static_top = _STATIC_VIEW_FUNCS[style](model, input_shape, legend=True, legend_position="top-left")
    static_bottom = _STATIC_VIEW_FUNCS[style](model, input_shape, legend=True, legend_position="bottom-left")
    frames = animate(model, input_shape, style=style, legend=True, legend_position="top-left")

    assert frames[-1].tobytes() == static_top.tobytes()
    assert frames[-1].tobytes() != static_bottom.tobytes()


@pytest.mark.parametrize("style", _STYLES)
def test_show_dimension_labels_appear_with_their_column(style: str, sequential_model: nn.Module) -> None:
    """A column's shape label appears exactly when that column's boxes do, not before."""
    input_shape = (1, 3, 8, 8)
    frames = animate(sequential_model, input_shape, style=style, show_dimension=True)
    no_labels_frames = animate(sequential_model, input_shape, style=style, show_dimension=False)

    # Every frame with labels should have at least as much ink as its no-label counterpart, and
    # strictly more once there's a column with a shape label reserved for it (frame 0 already
    # has a label for the input, so compare from frame 1 onward to catch a genuinely new label).
    for with_labels, without_labels in zip(frames[1:], no_labels_frames[1:], strict=True):
        assert _non_background_pixel_count(with_labels) > _non_background_pixel_count(without_labels)


@pytest.mark.parametrize("style", _STYLES)
def test_all_frames_share_canvas_size(style: str, residual_model: nn.Module) -> None:
    """Every frame must be the same size - only the content revealed changes, never the canvas."""
    frames = animate(residual_model, (1, 4, 8, 8), style=style)
    assert len({frame.size for frame in frames}) == 1


@pytest.mark.parametrize("style", _STYLES)
def test_to_file_writes_gif_and_returns_none(style: str, sequential_model: nn.Module, tmp_path: object) -> None:
    """Passing `to_file` writes the GIF and returns None, unlike the static `*_view()`."""
    out_path = tmp_path / "out.gif"
    result = animate(sequential_model, (1, 3, 8, 8), style=style, to_file=str(out_path))
    assert result is None
    assert out_path.exists()
    with Image.open(out_path) as saved:
        assert saved.format == "GIF"


@pytest.mark.parametrize("style", _STYLES)
def test_to_file_none_returns_frame_list(style: str, sequential_model: nn.Module) -> None:
    """`to_file=None` (the default) returns the raw per-column frames instead of writing a file."""
    result = animate(sequential_model, (1, 3, 8, 8), style=style, to_file=None)
    assert isinstance(result, list)
    assert all(isinstance(frame, Image.Image) for frame in result)


@pytest.mark.parametrize("style", _STYLES)
def test_frame_durations_and_loop_forever(style: str, sequential_model: nn.Module, tmp_path: object) -> None:
    """Written GIF frame durations and loop-forever metadata round-trip correctly."""
    out_path = tmp_path / "out.gif"
    animate(
        sequential_model,
        (1, 3, 8, 8),
        style=style,
        to_file=str(out_path),
        frame_duration=250,
        final_hold_duration=900,
        loop=True,
    )

    with Image.open(out_path) as saved:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(saved)]
        assert durations[:-1] == [250] * (len(durations) - 1)
        assert durations[-1] == 900
        assert saved.info.get("loop") == 0  # PIL's convention for "loop forever"


@pytest.mark.parametrize("style", _STYLES)
def test_loop_false_plays_once(style: str, sequential_model: nn.Module, tmp_path: object) -> None:
    """`loop=False` writes a GIF that plays once instead of looping forever."""
    out_path = tmp_path / "out.gif"
    animate(sequential_model, (1, 3, 8, 8), style=style, to_file=str(out_path), loop=False)
    with Image.open(out_path) as saved:
        assert saved.info.get("loop") == 1


@pytest.mark.parametrize("style", _STYLES)
def test_animate_multi_input_model_matches_static(style: str, siamese_model: nn.Module) -> None:
    """Parallel inputs reveal together (same frame), with no special-casing needed."""
    input_shape = ((1, 3, 8, 8), (1, 10))
    frames = animate(siamese_model, input_shape, style=style)
    static_img = _STATIC_VIEW_FUNCS[style](siamese_model, input_shape)
    assert frames[-1].tobytes() == static_img.tobytes()


@pytest.mark.parametrize("style", _STYLES)
def test_animate_multi_branch_model_matches_static(style: str, inception_model: nn.Module) -> None:
    """Parallel branches (not just a residual skip) reveal together, in the same frame."""
    input_shape = (1, 4, 8, 8)
    frames = animate(inception_model, input_shape, style=style)
    static_img = _STATIC_VIEW_FUNCS[style](inception_model, input_shape)
    assert frames[-1].tobytes() == static_img.tobytes()


def test_animate_defaults_to_graph_style(sequential_model: nn.Module) -> None:
    """Omitting style should animate the graph style (the default), matching render()'s default."""
    default_frames = animate(sequential_model, (1, 3, 8, 8))
    graph_frames = animate(sequential_model, (1, 3, 8, 8), style="graph")
    assert [f.tobytes() for f in default_frames] == [f.tobytes() for f in graph_frames]


def test_animate_rejects_unsupported_style(sequential_model: nn.Module) -> None:
    """An unrecognized style should raise a clear error, not silently fall back."""
    with pytest.raises(ValueError, match="Unsupported style"):
        animate(sequential_model, (1, 3, 8, 8), style="bogus")


def test_animate_rejects_typo_d_kwarg(sequential_model: nn.Module) -> None:
    """A typo'd kwarg should raise TypeError, not be silently ignored."""
    with pytest.raises(TypeError):
        animate(sequential_model, (1, 3, 8, 8), style="graph", nod_size=20)
