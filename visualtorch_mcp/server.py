"""MCP server exposing VisualTorch rendering tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import __version__
from .api_reference import docs_manifest
from .runner import render_model

mcp = FastMCP("visualtorch-mcp")


@mcp.tool()
def visualize_model(
    source: str,
    input_shape: Any,  # noqa: ANN401 - MCP accepts either one shape or per-input shapes.
    style: str = "graph",
    model_expression: str = "model",
    output_path: str | None = None,
    output_dir: str | None = None,
    options: dict[str, Any] | None = None,
    workdir: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Render a PyTorch model architecture diagram with VisualTorch.

    ``source`` should define the model and any classes it needs. By default the worker evaluates
    ``model_expression="model"`` after executing ``source``; set it to an expression such as
    ``Net()`` or ``build_model()`` when the source defines a class or factory instead.
    """
    return render_model(
        source=source,
        input_shape=input_shape,
        style=style,
        model_expression=model_expression,
        output_path=output_path,
        output_dir=output_dir,
        options=options,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def visualtorch_reference(style: str | None = None) -> str:
    """Return upstream VisualTorch docs links for styles and examples."""
    return docs_manifest(style)


@mcp.resource("visualtorch://docs")
def visualtorch_docs() -> str:
    """Return upstream VisualTorch documentation links."""
    return docs_manifest()


@mcp.resource("visualtorch://api-reference")
def visualtorch_api_reference() -> str:
    """Return the backward-compatible upstream API documentation resource."""
    return docs_manifest()


@mcp.resource("visualtorch://version")
def version() -> str:
    """Return the version of this MCP server."""
    return __version__


def main() -> None:
    """Run the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
