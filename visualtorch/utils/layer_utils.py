"""Layer Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

from .utils import get_keys_by_value

# WARNING: currently, graph visualization relying on following operations,
# thereby only linear and convolutional layers are supported.
# We need to implement a more dynamic/better approach to support all layers
TARGET_OPS = defaultdict(
    lambda: None,
    {"AddmmBackward0": nn.Linear, "ConvolutionBackward0": nn.Conv2d},
)


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
) -> tuple[dict[str, int], np.ndarray, list[list[torch.nn.Module]]]:
    """Extract adjacency matrix representation from a pytorch model.

    Args:
        model: PyTorch model.
        input_shape (tuple): The shape of the input tensor expected by the model, including batch dim.

    Returns:
        tuple: A tuple containing:
            - id_to_index_adj_mapping (dict): Mapping from node IDs to their corresponding index in
                the adjacency matrix.
            - adjacency_matrix (numpy.ndarray): The adjacency matrix representing connections between
                model operations/layers.
            - model_layers (list): List of model layers organized by their hierarchy.
    """
    dummy_input = torch.rand(input_shape)
    output_var = model(dummy_input)

    nodes: list = []
    edges: list = []
    id_to_ops: dict = {}

    max_level = [0]
    max_level_id = [""]

    # Extract nodes and edges for the target ops
    # Currently only the ones in the TARGET_OPS are supported.

    # handle multiple outputs
    if isinstance(output_var, tuple):
        for v in output_var:
            _add_base_tensor(v, id_to_ops, nodes, edges, max_level, max_level_id)
    else:
        _add_base_tensor(output_var, id_to_ops, nodes, edges, max_level, max_level_id)

    # Create adjacency matrix
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    id_to_index_adj_mapping = {node: idx for idx, node in enumerate(nodes)}

    for src_id, trg_id in edges:
        if trg_id is not None:
            src_index = id_to_index_adj_mapping[src_id]
            trg_index = id_to_index_adj_mapping[trg_id]
            adjacency_matrix[src_index, trg_index] += 1

    # Retrieve layers per level
    input_layer_id = max_level_id[0]
    temp_model_layers = [[input_layer_id]]

    while len(temp_model_layers) < max_level[0]:
        prev_layers = temp_model_layers[-1]
        new_layer = []
        for layer_id in prev_layers:
            src_index = id_to_index_adj_mapping[layer_id]
            for trg_idx in np.where(adjacency_matrix[src_index] > 0)[0]:
                trg_id = next(get_keys_by_value(id_to_index_adj_mapping, trg_idx))
                new_layer.append(trg_id)

        temp_model_layers.append(list(new_layer))

    # Filter duplicate layers
    seen = set()
    model_layers: list[list] = []
    for i in range(len(temp_model_layers) - 1, -1, -1):
        new_layers = []
        for layer_id in temp_model_layers[i]:
            if layer_id in seen:
                continue
            seen.add(layer_id)
            new_layers.append(id_to_ops[layer_id])
        model_layers.insert(0, list(new_layers))

    return id_to_index_adj_mapping, adjacency_matrix, model_layers


def add_input_dummy_layer(
    input_shape: tuple[int, ...],
    id_to_num_mapping: dict[str, int],
    adj_matrix: np.ndarray,
    model_layers: list[list[Any]],
) -> tuple[dict[str, int], np.ndarray, list[list[str]]]:
    """Add an input dummy layer to the model layers and update the adjacency matrix accordingly.

    Args:
        input_shape (tuple): The shape of the input tensor.
        id_to_num_mapping (dict): Mapping from node IDs to their corresponding index in the adjacency matrix.
        adj_matrix (numpy.ndarray): The adjacency matrix representing connections between model operations.
        model_layers (list): List of model layers organized by their dependencies.

    Returns:
        tuple: A tuple containing:
            - id_to_num_mapping (dict): Updated mapping from node IDs to their corresponding index in
                the adjacency matrix.
            - adj_matrix (numpy.ndarray): Updated adjacency matrix.
            - model_layers (list): Updated list of model layers with the input dummy layer.
    """
    first_hidden_layer = model_layers[0][0]
    input_dummy_layer = InputDummyLayer("input", input_shape[1])
    model_layers.insert(0, [input_dummy_layer])
    id_to_num_mapping[str(id(input_dummy_layer))] = len(id_to_num_mapping.keys())
    adj_matrix = np.pad(
        adj_matrix,
        ((0, 1), (0, 1)),
        mode="constant",
        constant_values=0,
    )
    adj_matrix[-1, id_to_num_mapping[str(id(first_hidden_layer))]] += 1
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
        if isinstance(out, tuple):
            if hasattr(out[0], "size"):
                layers[m_key]["output_shape"] = out[0].size()
            else:
                layers[m_key]["output_shape"] = tuple(o.size() for o in out if hasattr(o, "size"))
        elif isinstance(out, list):
            layers[m_key]["output_shape"] = tuple(o.size() for o in out if hasattr(o, "size"))
        else:
            layers[m_key]["output_shape"] = out.size()

    if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module is not model:
        hooks.append(module.register_forward_hook(hook))


def _add_nodes(
    fn: torch.autograd.function,
    id_to_ops: dict,
    nodes: list,
    edges: list,
    max_level: list[int],
    max_level_id: list[str],
    source: str | None = None,
    level: int = 0,
) -> None:
    assert not torch.is_tensor(fn)

    if str(type(fn).__name__) in TARGET_OPS:
        node_id = str(id(fn))
        id_to_ops[node_id] = fn
        if node_id not in nodes:
            nodes.append(node_id)
        level += 1
        if level > max_level[0]:
            max_level[0] = level
            max_level_id[0] = node_id

        edges.append((node_id, source))
        source = node_id

    # recurse
    if hasattr(fn, "next_functions"):
        for u in fn.next_functions:
            if u[0] is not None:
                _add_nodes(u[0], id_to_ops, nodes, edges, max_level, max_level_id, source, level)


def _add_base_tensor(
    var: torch.Tensor,
    id_to_ops: dict,
    nodes: list,
    edges: list,
    max_level: list[int],
    max_level_id: list[str],
) -> None:
    if var.grad_fn:
        _add_nodes(var.grad_fn, id_to_ops, nodes, edges, max_level, max_level_id)

    if var._is_view():  # noqa: SLF001
        _add_base_tensor(var._base, id_to_ops, nodes, edges, max_level, max_level_id)  # noqa: SLF001
