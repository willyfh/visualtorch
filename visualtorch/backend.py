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
from .utils.utils import InputShape, validate_input_shape

_INPUT_NODE_ID = "__input_dummy__"


@dataclass
class Architecture:
    """The structure of a traced model: layers grouped into columns, plus their adjacency.

    A column groups layers that sit at the same depth (longest path from the input) - a column
    holds more than one layer when the model has parallel branches. `columns[0]` always holds one
    or more synthetic input nodes (one per input tensor the model's forward() takes), so every
    real layer has at least one predecessor to connect from.
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


def extract_architecture(model: nn.Module, input_shape: InputShape) -> Architecture:
    """Trace a model's forward pass and extract its structure as an `Architecture`.

    Traces an actual forward pass (see `visualtorch.utils.recorder.trace_module_graph`) instead of
    walking the autograd backward graph, so any leaf module type is supported (not just a
    hardcoded list), and branching/skip-connections are captured naturally.

    Args:
        model: PyTorch model.
        input_shape (tuple): The shape of the input tensor expected by the model, including batch
            dim (e.g. (1, 3, 224, 224)). For a model whose forward() takes multiple separate input
            tensors, pass a tuple of per-tensor shapes instead, one per positional argument in
            order, e.g. ((1, 3, 224, 224), (1, 10)).

    Returns:
        Architecture: The traced model's structure.
    """
    input_shapes = validate_input_shape(input_shape)

    id_to_module, id_to_output_shape, edges, input_ids = trace_module_graph(model, input_shapes)

    nodes = list(id_to_module.keys())
    id_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}

    # Node ids that consume a traced input tensor directly, wired up below regardless of which
    # depth they end up at - a node can consume the raw input directly *and* have other
    # predecessors (e.g. the far side of a skip connection around a hidden sub-block). Keyed by
    # which input produced the edge, so each input's own box wires only to its own consumers.
    input_id_set = set(input_ids)
    direct_input_node_ids: dict[str, set[str]] = {input_id: set() for input_id in input_ids}
    adjacency = np.zeros((len(nodes), len(nodes)))
    successors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for src_id, trg_id in edges:
        if src_id in input_id_set:
            direct_input_node_ids[src_id].add(trg_id)
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

    id_to_index, adjacency, columns = _add_input_dummy_layers(
        input_shapes,
        input_ids,
        id_to_index,
        adjacency,
        columns,
        direct_input_node_ids,
    )

    id_to_column = {layer.node_id: col_idx for col_idx, column in enumerate(columns) for layer in column}

    return Architecture(columns=columns, adjacency=adjacency, id_to_index=id_to_index, id_to_column=id_to_column)


def _add_input_dummy_layers(
    input_shapes: tuple[tuple[int, ...], ...],
    input_ids: list[str],
    id_to_index: dict[str, int],
    adjacency: np.ndarray,
    columns: list[list[TracedLayer]],
    direct_input_node_ids: dict[str, set[str]],
) -> tuple[dict[str, int], np.ndarray, list[list[TracedLayer]]]:
    """Prepend one synthetic input node per input tensor, each wired to its own direct consumers."""
    n_inputs = len(input_shapes)
    input_dummy_node_ids = [f"{_INPUT_NODE_ID}#{i}" for i in range(n_inputs)]
    input_column = [
        TracedLayer(
            module=InputDummyLayer(_input_label(i, n_inputs), shape[1] if len(shape) > 1 else None),
            output_shape=shape,
            node_id=input_dummy_node_ids[i],
        )
        for i, shape in enumerate(input_shapes)
    ]
    columns.insert(0, input_column)

    for node_id in input_dummy_node_ids:
        id_to_index[node_id] = len(id_to_index)
    adjacency = np.pad(adjacency, ((0, n_inputs), (0, n_inputs)), mode="constant", constant_values=0)

    for i, input_id in enumerate(input_ids):
        input_index = id_to_index[input_dummy_node_ids[i]]
        for node_id in direct_input_node_ids[input_id]:
            adjacency[input_index, id_to_index[node_id]] += 1

    return id_to_index, adjacency, columns


def _input_label(index: int, n_inputs: int) -> str:
    """Label a synthetic input node - unchanged "input" for a single-input model, else indexed."""
    return "input" if n_inputs == 1 else f"input_{index}"
