"""Layer Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import warnings


class Input:
    """A placeholder standing in for the raw input tensor, before any real computation."""

    def __init__(self, name: str, units: int | None = None) -> None:
        if units:
            self.units = units
        self._name = name

    def name(self) -> str:
        """Return layer name"""
        return self._name


class Output:
    """A placeholder standing in for a final merge (e.g. a residual add) that nothing consumes.

    The tracer only records an edge when a tensor is passed into another leaf module call. A
    tensor merge (e.g. `branch + shortcut`) whose result is the model's literal return value,
    with no subsequent module call to receive it, would otherwise silently vanish - not just
    losing a "this is where they combine" indicator, but dropping the shortcut branch's edge
    from the graph entirely. This placeholder gives those final producers somewhere to connect.
    """

    def __init__(self, name: str, units: int | None = None) -> None:
        if units:
            self.units = units
        self._name = name

    def name(self) -> str:
        """Return layer name"""
        return self._name


def __getattr__(name: str) -> object:
    """Lazily provide `InputDummyLayer`, the pre-1.1 name for `Input`, with a deprecation warning."""
    if name == "InputDummyLayer":
        warnings.warn(
            "`InputDummyLayer` is deprecated and will be removed in a future release, use `Input` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Input
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
