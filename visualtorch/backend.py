"""Backend module for pytorch model visualization.

Single entry point for extracting a model's structure - topology and shapes - via the traced
adjacency graph mechanism (`visualtorch.utils.recorder`), for every rendering frontend to consume.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from torch import nn

from .utils.layer_utils import InputDummyLayer
from .utils.recorder import trace_module_graph
from .utils.traced_layer import TracedLayer
from .utils.utils import validate_input_shape

_INPUT_NODE_ID = "__input_dummy__"


@dataclass
class Architecture:
    """The structure of a traced model: layers grouped into columns, plus their adjacency.

    A column groups layers that sit at the same depth (longest path from the input) - a column
    holds more than one layer when the model has parallel branches. `columns[0]` is always a
    synthetic input node, so every real layer has at least one predecessor to connect from.
    """

    columns: list[list[TracedLayer]]
    adjacency: np.ndarray
    id_to_index: dict[str, int]
    id_to_column: dict[str, int]

    @cached_property
    def _index_to_id(self) -> dict[int, str]:
        return {index: node_id for node_id, index in self.id_to_index.items()}

    def edges(self) -> Iterator[tuple[str, str]]:
        """Yield every `(start_id, end_id)` edge present in the adjacency matrix."""
        for start_idx, end_idx in zip(*np.where(self.adjacency > 0), strict=False):
            yield self._index_to_id[int(start_idx)], self._index_to_id[int(end_idx)]


def extract_architecture(model: nn.Module, input_shape: tuple[int, ...]) -> Architecture:
    """Trace a model's forward pass and extract its structure as an `Architecture`.

    Traces an actual forward pass (see `visualtorch.utils.recorder.trace_module_graph`) instead of
    walking the autograd backward graph, so any leaf module type is supported (not just a
    hardcoded list), and branching/skip-connections are captured naturally.

    Args:
        model: PyTorch model.
        input_shape (tuple): The shape of the input tensor expected by the model, including batch dim.

    Returns:
        Architecture: The traced model's structure.
    """
    validate_input_shape(input_shape)

    id_to_module, id_to_output_shape, edges, input_id = trace_module_graph(model, input_shape)

    nodes = list(id_to_module.keys())
    id_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}

    # Node ids that consume the traced input tensor directly, wired up below regardless of which
    # depth they end up at - a node can consume the raw input directly *and* have other
    # predecessors (e.g. the far side of a skip connection around a hidden sub-block).
    direct_input_node_ids: set[str] = set()
    adjacency = np.zeros((len(nodes), len(nodes)))
    successors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for src_id, trg_id in edges:
        if src_id == input_id:
            direct_input_node_ids.add(trg_id)
            continue
        adjacency[id_to_index[src_id], id_to_index[trg_id]] += 1
        successors[src_id].append(trg_id)
        in_degree[trg_id] += 1

    # Longest-path-from-source layering: a node's depth is 1 + the max depth of its predecessors
    # (0 if it has none), computed via topological order so a merge node (e.g. the far side of a
    # skip connection) is never placed before a branch that hasn't converged yet.
    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for node_id in nodes:
        if in_degree[node_id] == 0:
            depth[node_id] = 0
            queue.append(node_id)

    while queue:
        node_id = queue.popleft()
        for succ_id in successors[node_id]:
            depth[succ_id] = max(depth.get(succ_id, 0), depth[node_id] + 1)
            in_degree[succ_id] -= 1
            if in_degree[succ_id] == 0:
                queue.append(succ_id)

    max_depth = max(depth.values(), default=-1)
    columns: list[list[TracedLayer]] = [[] for _ in range(max_depth + 1)]
    for node_id in nodes:
        wrapper = TracedLayer(
            module=id_to_module[node_id],
            output_shape=id_to_output_shape[node_id],
            node_id=node_id,
        )
        columns[depth[node_id]].append(wrapper)

    id_to_index, adjacency, columns = _add_input_dummy_layer(
        input_shape,
        id_to_index,
        adjacency,
        columns,
        direct_input_node_ids,
    )

    id_to_column = {layer.node_id: col_idx for col_idx, column in enumerate(columns) for layer in column}

    return Architecture(columns=columns, adjacency=adjacency, id_to_index=id_to_index, id_to_column=id_to_column)


def _add_input_dummy_layer(
    input_shape: tuple[int, ...],
    id_to_index: dict[str, int],
    adjacency: np.ndarray,
    columns: list[list[TracedLayer]],
    direct_input_node_ids: set[str],
) -> tuple[dict[str, int], np.ndarray, list[list[TracedLayer]]]:
    """Prepend a synthetic input node and wire it up to every direct consumer of the input."""
    input_dummy_layer = TracedLayer(
        module=InputDummyLayer("input", input_shape[1]),
        output_shape=input_shape,
        node_id=_INPUT_NODE_ID,
    )
    columns.insert(0, [input_dummy_layer])
    id_to_index[_INPUT_NODE_ID] = len(id_to_index)
    adjacency = np.pad(adjacency, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    input_index = id_to_index[_INPUT_NODE_ID]
    for node_id in direct_input_node_ids:
        adjacency[input_index, id_to_index[node_id]] += 1
    return id_to_index, adjacency, columns
