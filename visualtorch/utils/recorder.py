"""Generic module-graph tracing for pytorch model visualization.

Traces a forward pass to recover, for every leaf module actually invoked, which other leaf
module(s) produced its input(s) - without hardcoding supported layer types and without relying
on private autograd attributes. Adapted from torchview's (github.com/mert-kurttutan/torchview)
tensor-subclassing approach, stripped down to only what graph_view needs: one node per leaf
module (mirroring the leaf definition already used by register_hook), not a node per tensor op.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

INPUT_NODE_ID = "__input__"

_orig_module_call = nn.Module.__call__


class RecorderTensor(torch.Tensor):
    """A torch.Tensor subclass that propagates which leaf module(s) produced its data.

    Carries a `_producer_ids` set (module ids, `str(id(module))`) through arbitrary tensor
    operations via `__torch_function__`, so that when a tensor eventually reaches another leaf
    module call, we know exactly which upstream leaf module(s) it came from - including across
    ops like `add`/`cat` that merge multiple branches (e.g. residual/skip connections).
    """

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Intercept every tensor op to propagate producer ids onto the output."""
        if kwargs is None:
            kwargs = {}

        producer_ids = _collect_producer_ids(args) | _collect_producer_ids(kwargs)

        out = super().__torch_function__(func, types, args, kwargs)

        if producer_ids:
            _stamp_producer_ids(out, producer_ids)

        return out


def _collect_producer_ids(obj: Any) -> set[str]:
    """Recursively collect producer ids from any RecorderTensor found inside obj."""
    ids: set[str] = set()
    if isinstance(obj, RecorderTensor):
        ids.update(getattr(obj, "_producer_ids", set()))
    elif isinstance(obj, torch.Tensor):
        pass
    elif isinstance(obj, Mapping):
        for value in obj.values():
            ids.update(_collect_producer_ids(value))
    elif isinstance(obj, list | tuple | set):
        for value in obj:
            ids.update(_collect_producer_ids(value))
    return ids


def _stamp_producer_ids(obj: Any, ids: set[str]) -> None:
    """Recursively stamp producer ids onto any RecorderTensor found inside obj."""
    if isinstance(obj, RecorderTensor):
        obj._producer_ids = set(ids)  # noqa: SLF001
    elif isinstance(obj, torch.Tensor):
        pass
    elif isinstance(obj, Mapping):
        for value in obj.values():
            _stamp_producer_ids(value, ids)
    elif isinstance(obj, list | tuple | set):
        for value in obj:
            _stamp_producer_ids(value, ids)


def _wrap_and_stamp(obj: Any, ids: set[str]) -> Any:
    """Like _stamp_producer_ids, but also converts plain (non-RecorderTensor) tensors.

    A leaf module's forward can internally escape the tensor subclass (e.g. a numpy
    round-trip), which would otherwise silently break lineage tracking for everything
    downstream of that module. Re-wrapping at the module-call boundary re-establishes it
    regardless of what happened inside.
    """
    if isinstance(obj, RecorderTensor):
        obj._producer_ids = set(ids)  # noqa: SLF001
        return obj
    if isinstance(obj, torch.Tensor):
        wrapped = obj.as_subclass(RecorderTensor)
        wrapped._producer_ids = set(ids)  # noqa: SLF001
        return wrapped
    if isinstance(obj, Mapping):
        return type(obj)({key: _wrap_and_stamp(value, ids) for key, value in obj.items()})  # type: ignore[call-arg]
    if isinstance(obj, tuple):
        return type(obj)(_wrap_and_stamp(value, ids) for value in obj)  # type: ignore[call-arg]
    if isinstance(obj, list):
        return [_wrap_and_stamp(value, ids) for value in obj]
    return obj


def _first_tensor_shape(obj: Any) -> tuple[int, ...]:
    """Recursively find the shape of the first tensor inside obj, or () if there isn't one."""
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, Mapping):
        for value in obj.values():
            shape = _first_tensor_shape(value)
            if shape:
                return shape
        return ()
    if isinstance(obj, list | tuple):
        for value in obj:
            shape = _first_tensor_shape(value)
            if shape:
                return shape
        return ()
    return ()


def _wrapped_module_call(
    id_to_module: dict[str, nn.Module],
    id_to_output_shape: dict[str, tuple[int, ...]],
    edges: list[tuple[str, str]],
    call_counts: dict[int, int],
) -> Any:
    """Build the replacement for nn.Module.__call__ used while tracing."""

    def wrapped(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        producer_ids = _collect_producer_ids(args) | _collect_producer_ids(kwargs)

        out = _orig_module_call(mod, *args, **kwargs)

        # Only leaf modules (no children) become graph nodes - containers such as
        # nn.Sequential or a custom composite block are transparent plumbing, matching the
        # leaf definition already used by register_hook for layered_view/lenet_view. A leaf
        # module invoked more than once (e.g. weight sharing) gets a distinct node per call,
        # keyed by a call-index suffix, so repeat calls don't collapse into a self-referential
        # node that can never be placed in the topological layering.
        is_leaf = len(list(mod.children())) == 0
        if is_leaf:
            base_id = id(mod)
            call_index = call_counts.get(base_id, 0)
            call_counts[base_id] = call_index + 1
            node_id = f"{base_id}#{call_index}"

            id_to_module[node_id] = mod
            id_to_output_shape[node_id] = _first_tensor_shape(out)
            edges.extend((producer_id, node_id) for producer_id in producer_ids)
            out = _wrap_and_stamp(out, {node_id})

        return out

    return wrapped


class Recorder:
    """Context manager that temporarily wraps nn.Module.__call__ to trace leaf-module calls."""

    def __init__(
        self,
        id_to_module: dict[str, nn.Module],
        id_to_output_shape: dict[str, tuple[int, ...]],
        edges: list[tuple[str, str]],
        call_counts: dict[int, int],
    ) -> None:
        self._id_to_module = id_to_module
        self._id_to_output_shape = id_to_output_shape
        self._edges = edges
        self._call_counts = call_counts

    def __enter__(self) -> "Recorder":
        """Patch nn.Module.__call__ to start tracing."""
        nn.Module.__call__ = _wrapped_module_call(  # type: ignore[method-assign]
            self._id_to_module,
            self._id_to_output_shape,
            self._edges,
            self._call_counts,
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        """Restore the original nn.Module.__call__."""
        nn.Module.__call__ = _orig_module_call  # type: ignore[method-assign]


def trace_module_graph(
    model: nn.Module,
    input_shape: tuple[int, ...],
) -> tuple[dict[str, nn.Module], dict[str, tuple[int, ...]], list[tuple[str, str]], str]:
    """Trace a forward pass to recover the leaf-module call graph.

    Args:
        model (nn.Module): The model to trace.
        input_shape (tuple): The shape of the dummy input tensor, including batch dim.

    Returns:
        tuple: A tuple containing:
            - id_to_module (dict): Mapping from node id to the leaf module. A leaf module
                called more than once gets one entry per call (`f"{id(module)}#{call_index}"`).
            - id_to_output_shape (dict): Mapping from node id to that module's output shape.
            - edges (list): `(producer_node_id, consumer_node_id)` pairs, in call order.
            - input_id (str): The synthetic node id representing the traced input tensor.
    """
    dummy_input = torch.rand(input_shape).as_subclass(RecorderTensor)
    dummy_input._producer_ids = {INPUT_NODE_ID}  # noqa: SLF001

    id_to_module: dict[str, nn.Module] = {}
    id_to_output_shape: dict[str, tuple[int, ...]] = {}
    edges: list[tuple[str, str]] = []
    call_counts: dict[int, int] = {}

    with Recorder(id_to_module, id_to_output_shape, edges, call_counts):
        model(dummy_input)

    return id_to_module, id_to_output_shape, edges, INPUT_NODE_ID
