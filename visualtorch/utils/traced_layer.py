"""Traced layer wrapper for graph-style pytorch model visualization."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from torch import nn


@dataclass
class TracedLayer:
    """A single leaf-module node recovered while tracing a model's forward pass.

    `node_id` must be the same string key used to index this node in the adjacency matrix
    (see `visualtorch.utils.recorder.trace_module_graph`), not `id()` of this wrapper - otherwise
    connector drawing in graph.py silently fails to find the node it should connect to.
    """

    module: nn.Module
    output_shape: tuple[int, ...]
    node_id: str
    extra_output_shapes: tuple[tuple[int, ...], ...] = ()
    """Shapes of any additional output tensors beyond `output_shape` (e.g. `nn.LSTM`'s hidden and
    cell state, returned alongside its main sequence output) - empty for a module that returns
    just one tensor. `output_shape` alone still drives box sizing; this is only ever read to
    extend the `show_dimension` label so those extra tensors aren't silently unaccounted for.
    """
