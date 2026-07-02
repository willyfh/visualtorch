"""Layer Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT


class InputDummyLayer:
    """A dummy layer for input."""

    def __init__(self, name: str, units: int | None = None) -> None:
        if units:
            self.units = units
        self._name = name

    def name(self) -> str:
        """Return layer name"""
        return self._name
