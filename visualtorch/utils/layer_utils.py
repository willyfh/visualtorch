"""Layer Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import OrderedDict

import torch
from torch import nn


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
