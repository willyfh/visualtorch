import aggdraw
from PIL import Image
from math import ceil
from .layer_utils import model_to_adj_matrix, add_input_dummy_layer
from .utils import Circle, Ellipses, get_keys_by_value, Box
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import torch


def graph_view(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    to_file: Optional[str] = None,
    color_map: Optional[Dict[Any, Any]] = None,
    node_size: int = 50,
    background_fill: Any = "white",
    padding: int = 10,
    layer_spacing: int = 250,
    node_spacing: int = 10,
    connector_fill: Any = "gray",
    connector_width: int = 1,
    ellipsize_after: int = 10,
    inout_as_tensor: bool = True,
    show_neurons: bool = True,
) -> Image.Image:
    """
    Generates an architecture visualization for a given linear PyTorch model (i.e., one input and output tensor for each
    layer) in a graph style.

    Args:
        model (torch.nn.Module): A PyTorch model that will be visualized.
        input_shape (tuple): The shape of the input tensor.
        to_file (str, optional): Path to the file to write the created image to. If the image does not exist yet,
            it will be created, else overwritten. Image type is inferred from the file ending. Providing None
            will disable writing.
        color_map (dict, optional): Dict defining fill and outline for each layer by class type. Will fallback to default
            values for not specified classes.
        node_size (int, optional): Size in pixels each node will have.
        background_fill (Any, optional): Color for the image background. Can be str or (R,G,B,A).
        padding (int, optional): Distance in pixels before the first and after the last layer.
        layer_spacing (int, optional): Spacing in pixels between two layers.
        node_spacing (int, optional): Spacing in pixels between nodes.
        connector_fill (Any, optional): Color for the connectors. Can be str or (R,G,B,A).
        connector_width (int, optional): Line-width of the connectors in pixels.
        ellipsize_after (int, optional): Maximum number of neurons per layer to draw. If a layer is exceeding this,
            the remaining neurons will be drawn as ellipses.
        inout_as_tensor (bool, optional): If True there will be one input and output node for each tensor, else the
            tensor will be flattened and one node for each scalar will be created (e.g., a (10, 10) shape will be
            represented by 100 nodes).
        show_neurons (bool, optional): If True a node for each neuron in supported layers is created (constrained by
            ellipsize_after), else each layer is represented by a node.

    Returns:
        Image.Image: Generated architecture image.
    """

    if color_map is None:
        color_map = dict()

    # Iterate over the model to compute bounds and generate boxes

    layers: List[Any] = list()
    layer_y = list()

    # Attach helper layers

    id_to_num_mapping, adj_matrix, model_layers = model_to_adj_matrix(
        model, input_shape
    )

    # Add fake input layers

    id_to_num_mapping, adj_matrix, model_layers = add_input_dummy_layer(
        input_shape, id_to_num_mapping, adj_matrix, model_layers
    )

    # Create architecture

    current_x = padding  # + input_label_size[0] + text_padding

    id_to_node_list_map = dict()

    for index, layer_list in enumerate(model_layers):
        current_y = 0
        nodes = []
        for layer in layer_list:
            is_box = True
            units = 1

            if show_neurons:
                if hasattr(layer, "_saved_bias_sym_sizes_opt"):
                    is_box = False
                    units = layer._saved_bias_sym_sizes_opt[0]
                elif hasattr(layer, "_saved_mat2_sym_sizes"):
                    is_box = False
                    units = layer._saved_mat2_sym_sizes[1]
                elif hasattr(layer, "units"):  # for dummy input layer
                    is_box = False
                    units = layer.units

            n = min(units, ellipsize_after)
            layer_nodes = list()

            for i in range(n):
                scale = 1
                c: Box | Circle | Ellipses
                if not is_box:
                    if i != ellipsize_after - 2:
                        c = Circle()
                    else:
                        c = Ellipses()
                else:
                    c = Box()
                    scale = 3

                c.x1 = current_x
                c.y1 = current_y
                c.x2 = c.x1 + node_size
                c.y2 = c.y1 + node_size * scale

                current_y = c.y2 + node_spacing

                c.fill = color_map.get(type(layer), {}).get("fill", "#ADD8E6")
                c.outline = color_map.get(type(layer), {}).get("outline", "black")

                layer_nodes.append(c)

            id_to_node_list_map[str(id(layer))] = layer_nodes
            nodes.extend(layer_nodes)
            current_y += 2 * node_size

        layer_y.append(current_y - node_spacing - 2 * node_size)
        layers.append(nodes)
        current_x += node_size + layer_spacing

    # Generate image

    img_width = (
        len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding
    )
    img_height = max(*layer_y) + 2 * padding
    img = Image.new(
        "RGBA", (int(ceil(img_width)), int(ceil(img_height))), background_fill
    )

    draw = aggdraw.Draw(img)

    # y correction (centering)
    for i, layer in enumerate(layers):
        y_off = (img.height - layer_y[i]) / 2
        node: Any
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off

    for start_idx, end_idx in zip(*np.where(adj_matrix > 0)):
        start_id = next(get_keys_by_value(id_to_num_mapping, start_idx))
        end_id = next(get_keys_by_value(id_to_num_mapping, end_idx))

        start_layer_list = id_to_node_list_map[start_id]
        end_layer_list = id_to_node_list_map[end_id]

        # draw connectors
        for start_node_idx, start_node in enumerate(start_layer_list):
            for end_node in end_layer_list:
                if not isinstance(start_node, Ellipses) and not isinstance(
                    end_node, Ellipses
                ):
                    _draw_connector(
                        draw,
                        start_node,
                        end_node,
                        color=connector_fill,
                        width=connector_width,
                    )

    for i, layer in enumerate(layers):
        for node_index, node in enumerate(layer):
            node.draw(draw)

    draw.flush()

    if to_file is not None:
        img.save(to_file)

    return img


def _draw_connector(draw, start_node, end_node, color, width):
    pen = aggdraw.Pen(color, width)
    x1 = start_node.x2
    y1 = start_node.y1 + (start_node.y2 - start_node.y1) / 2
    x2 = end_node.x1
    y2 = end_node.y1 + (end_node.y2 - end_node.y1) / 2
    draw.line([x1, y1, x2, y2], pen)
