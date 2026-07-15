"""Modules for pytorch model visualization."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

import warnings

from visualtorch.flow import layered_view  # noqa: F401 - deprecated re-export, not in __all__
from visualtorch.render import (
    CommonOptions,
    FlowStyleOptions,
    GraphStyleOptions,
    LenetStyleOptions,
    animate,
    render,
)
from visualtorch.utils.layer_utils import Input
from visualtorch.utils.utils import PALETTES

__all__ = [
    "PALETTES",
    "CommonOptions",
    "FlowStyleOptions",
    "GraphStyleOptions",
    "Input",
    "LenetStyleOptions",
    "animate",
    "render",
]


def __getattr__(name: str) -> object:
    """Lazily provide `InputDummyLayer`, the pre-1.1 name for `Input`, with a deprecation warning."""
    if name == "InputDummyLayer":
        warnings.warn(
            "`visualtorch.InputDummyLayer` is deprecated and will be removed in a future "
            "release, use `visualtorch.Input` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Input
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
