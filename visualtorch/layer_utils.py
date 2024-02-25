import numpy as np

import torch
import torch.nn as nn

from .utils import get_keys_by_value

from typing import Tuple, Dict, List, Any


TARGET_OPS = {"AddmmBackward0", "ConvolutionBackward0"}


class SpacingDummyLayer(nn.Module):
    def __init__(self, spacing: int = 50):
        super().__init__()
        self.spacing = spacing


class InputDummyLayer:
    def __init__(self, name, units=None):
        if units:
            self.units = units
        self.name = name


def model_to_adj_matrix(
    model, input_shape
) -> Tuple[Dict[str, int], np.ndarray, List[List[torch.nn.Module]]]:
    """
    Extract adjacency matrix representation from a pytorch model.

    Args:
        model: PyTorch model.
        input_shape (tuple): The shape of the input tensor expected by the model, including batch dim.

    Returns:
        tuple: A tuple containing:
            - id_to_index_adj_mapping (dict): Mapping from node IDs to their corresponding index in the adjacency matrix.
            - adjacency_matrix (numpy.ndarray): The adjacency matrix representing connections between model operations/layers.
            - model_layers (list): List of model layers organized by their hierarchy.
    """
    dummy_input = torch.rand(input_shape)
    output_var = model(dummy_input)

    nodes = []
    edges = []
    id_to_ops = {}

    max_level = [0]
    max_level_id = [""]

    def add_nodes(fn, source=None, level=0):
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
                    add_nodes(u[0], source, level)

    def add_base_tensor(var):
        if var.grad_fn:
            add_nodes(var.grad_fn)

        if var._is_view():
            add_base_tensor(var._base)

    # Extract nodes and edges for the target ops
    # Currently only the ones in the TARGET_OPS are supported.

    # handle multiple outputs
    if isinstance(output_var, tuple):
        for v in output_var:
            add_base_tensor(v)
    else:
        add_base_tensor(output_var)

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
    model_layers: List[List] = []
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
    input_shape: Tuple[int, ...],
    id_to_num_mapping: Dict[str, int],
    adj_matrix: np.ndarray,
    model_layers: List[List[Any]],
) -> Tuple[Dict[str, int], np.ndarray, List[List[str]]]:
    """
    Add an input dummy layer to the model layers and update the adjacency matrix accordingly.

    Args:
        input_shape (tuple): The shape of the input tensor.
        id_to_num_mapping (dict): Mapping from node IDs to their corresponding index in the adjacency matrix.
        adj_matrix (numpy.ndarray): The adjacency matrix representing connections between model operations.
        model_layers (list): List of model layers organized by their dependencies.

    Returns:
        tuple: A tuple containing:
            - id_to_num_mapping (dict): Updated mapping from node IDs to their corresponding index in the adjacency matrix.
            - adj_matrix (numpy.ndarray): Updated adjacency matrix.
            - model_layers (list): Updated list of model layers with the input dummy layer.
    """
    first_hidden_layer = model_layers[0][0]
    input_dummy_layer = InputDummyLayer("input", input_shape[1])
    model_layers.insert(0, [input_dummy_layer])
    id_to_num_mapping[str(id(input_dummy_layer))] = len(id_to_num_mapping.keys())
    adj_matrix = np.pad(
        adj_matrix, ((0, 1), (0, 1)), mode="constant", constant_values=0
    )
    adj_matrix[-1, id_to_num_mapping[str(id(first_hidden_layer))]] += 1
    return id_to_num_mapping, adj_matrix, model_layers
