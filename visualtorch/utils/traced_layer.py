"""Traced layer wrapper for graph-style pytorch model visualization."""

# Copyright (C) 2024 Willy Fitra Hendria
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
