"""Layer Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import OrderedDict, deque

import numpy as np
import torch
from torch import nn

from .recorder import trace_module_graph
from .traced_layer import TracedLayer


class SpacingDummyLayer(nn.Module):
    """A dummy layer to add spacing."""

    def __init__(self, spacing: int = 50) -> None:
        super().__init__()
        self.spacing = spacing


class InputDummyLayer:
    """A dummy layer for input."""

    def __init__(self, name: str, units: int | None = None) -> None:
        if units:
            self.units = units
        self._name = name

    def name(self) -> str:
        """Return layer name"""
        return self._name


def model_to_adj_matrix(
    model: nn.Module | nn.Sequential,
    input_shape: tuple[int, ...],
) -> tuple[dict[str, int], np.ndarray, list[list[TracedLayer]]]:
    """Extract adjacency matrix representation from a pytorch model.

    Traces an actual forward pass (see `visualtorch.utils.recorder.trace_module_graph`) instead of
    walking the autograd backward graph, so any leaf module type is supported (not just a
    hardcoded list), and branching/skip-connections are captured naturally.

    Args:
        model: PyTorch model.
        input_shape (tuple): The shape of the input tensor expected by the model, including batch dim.

    Returns:
        tuple: A tuple containing:
            - id_to_index_adj_mapping (dict): Mapping from node IDs to their corresponding index in
                the adjacency matrix.
            - adjacency_matrix (numpy.ndarray): The adjacency matrix representing connections between
                model layers.
            - model_layers (list): List of `TracedLayer` wrappers organized by their hierarchy.
    """
    id_to_module, id_to_output_shape, edges, input_id = trace_module_graph(model, input_shape)

    nodes = list(id_to_module.keys())
    id_to_index_adj_mapping = {node_id: idx for idx, node_id in enumerate(nodes)}

    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    successors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for src_id, trg_id in edges:
        if src_id == input_id:
            continue
        adjacency_matrix[id_to_index_adj_mapping[src_id], id_to_index_adj_mapping[trg_id]] += 1
        successors[src_id].append(trg_id)
        in_degree[trg_id] += 1

    # Longest-path-from-source layering: a node's depth is 1 + the max depth of its
    # predecessors (0 if it has none), computed via topological order so a merge node
    # (e.g. the far side of a skip connection) is never placed before a branch that
    # hasn't converged yet.
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
    model_layers: list[list[TracedLayer]] = [[] for _ in range(max_depth + 1)]
    for node_id in nodes:
        wrapper = TracedLayer(
            module=id_to_module[node_id],
            output_shape=id_to_output_shape[node_id],
            node_id=node_id,
        )
        model_layers[depth[node_id]].append(wrapper)

    return id_to_index_adj_mapping, adjacency_matrix, model_layers


def add_input_dummy_layer(
    input_shape: tuple[int, ...],
    id_to_num_mapping: dict[str, int],
    adj_matrix: np.ndarray,
    model_layers: list[list[TracedLayer]],
) -> tuple[dict[str, int], np.ndarray, list[list[TracedLayer]]]:
    """Add an input dummy layer to the model layers and update the adjacency matrix accordingly.

    Args:
        input_shape (tuple): The shape of the input tensor.
        id_to_num_mapping (dict): Mapping from node IDs to their corresponding index in the adjacency matrix.
        adj_matrix (numpy.ndarray): The adjacency matrix representing connections between model layers.
        model_layers (list): List of `TracedLayer` wrappers organized by their dependencies.

    Returns:
        tuple: A tuple containing:
            - id_to_num_mapping (dict): Updated mapping from node IDs to their corresponding index in
                the adjacency matrix.
            - adj_matrix (numpy.ndarray): Updated adjacency matrix.
            - model_layers (list): Updated list of model layers with the input dummy layer.
    """
    first_layer = model_layers[0]
    input_node_id = "__input_dummy__"
    input_dummy_layer = TracedLayer(
        module=InputDummyLayer("input", input_shape[1]),
        output_shape=input_shape,
        node_id=input_node_id,
    )
    model_layers.insert(0, [input_dummy_layer])
    id_to_num_mapping[input_node_id] = len(id_to_num_mapping.keys())
    adj_matrix = np.pad(
        adj_matrix,
        ((0, 1), (0, 1)),
        mode="constant",
        constant_values=0,
    )
    input_index = id_to_num_mapping[input_node_id]
    for layer in first_layer:
        adj_matrix[input_index, id_to_num_mapping[layer.node_id]] += 1
    return id_to_num_mapping, adj_matrix, model_layers


def register_hook(
    model: nn.Module,
    module: nn.Module,
    hooks: list,
    layers: OrderedDict,
) -> None:
    """Registers a forward hook on the specified module and collects the module and the output shapes.

    Args:
        model (nn.Module): The parent model.
        module (nn.Module): The module to register the hook on.
        hooks (List): A list to store the registered hooks.
        layers (OrderedDict): An OrderedDict to store information about the registered modules and output shapes.

    Returns:
        None
    """

    def hook(
        module: nn.Module,
        _: tuple[torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(layers)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        layers[m_key] = OrderedDict()
        layers[m_key]["module"] = module
        if isinstance(out, tuple | list):
            if len(out) > 0 and hasattr(out[0], "size"):
                layers[m_key]["output_shape"] = out[0].size()
            else:
                layers[m_key]["output_shape"] = tuple(o.size() for o in out if hasattr(o, "size"))
        else:
            layers[m_key]["output_shape"] = out.size()

    # Only hook leaf modules (no children). Container modules - whether nn.Sequential,
    # nn.ModuleList, or a custom container such as timm's FeatureListNet - would otherwise be
    # captured as if they were a single layer, with their multi-tensor output mistaken for one
    # layer's output shape.
    is_leaf = len(list(module.children())) == 0
    if is_leaf and module is not model:
        hooks.append(module.register_forward_hook(hook))
