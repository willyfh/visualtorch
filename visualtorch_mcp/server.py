"""Client-neutral MCP server exposing VisualTorch rendering tools."""

from __future__ import annotations

import argparse
import json
from functools import partial
from threading import Event
from typing import TYPE_CHECKING, Annotated

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from . import MCP_API_VERSION, __version__
from .api_reference import capabilities_manifest, docs_manifest
from .runner import animate_model as run_animation
from .runner import render_model

if TYPE_CHECKING:
    from collections.abc import Callable

SERVER_INSTRUCTIONS = """
VisualTorch MCP renders architecture diagrams from trusted local Python source.

Use visualize_model for a static PNG and animate_model for an animated GIF. The source argument
is executed as Python in a child process and must define the model or a model factory referenced by
model_expression. This process boundary is not a security sandbox: source inherits the local user's
filesystem, environment, and network permissions. Never submit source you do not trust.

Both rendering tools write an artifact to output_path (or a generated path under output_dir) and
return structured file metadata. Call visualtorch_capabilities before rendering to discover styles,
aliases, palettes, style-specific options, animation timing options, and output contracts.
""".strip()

Dimension = Annotated[int, Field(gt=0, description="A positive tensor dimension.")]
InputShape = list[Dimension] | list[list[Dimension]] | str
StyleName = Annotated[
    str,
    Field(
        description=(
            "VisualTorch style or alias; names are case-insensitive and hyphens are accepted. "
            "Call visualtorch_capabilities for the current list."
        ),
    ),
]
TimeoutSeconds = Annotated[int, Field(ge=1, le=600, description="Worker timeout in seconds.")]

mcp = FastMCP(
    "visualtorch-mcp",
    instructions=SERVER_INSTRUCTIONS,
)


@mcp.tool()
async def visualize_model(
    source: str,
    input_shape: InputShape,
    style: StyleName = "graph",
    model_expression: str = "model",
    output_path: str | None = None,
    output_dir: str | None = None,
    options: dict[str, object] | None = None,
    workdir: str | None = None,
    timeout_seconds: TimeoutSeconds = 120,
) -> dict[str, object]:
    """Render a trusted PyTorch model source as a static VisualTorch PNG.

    The source must define the model and any classes it needs. By default the worker evaluates
    ``model_expression="model"`` after executing source; use an expression such as ``Net()`` or
    ``build_model()`` when source defines a class or factory. Input shape includes the batch
    dimension and may be one shape or a list of shapes for a multi-input model.

    This tool executes source code with the current user's permissions. It is process-separated
    for reliability, not sandboxed for security, so only provide source you trust. The returned
    object describes the PNG written to disk; it does not embed the image bytes.
    """
    return await _run_cancellable_job(
        render_model,
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
async def animate_model(
    source: str,
    input_shape: InputShape,
    style: StyleName = "graph",
    model_expression: str = "model",
    output_path: str | None = None,
    output_dir: str | None = None,
    options: dict[str, object] | None = None,
    workdir: str | None = None,
    timeout_seconds: TimeoutSeconds = 120,
) -> dict[str, object]:
    """Render a trusted PyTorch model source as an animated VisualTorch GIF.

    The model-source and input-shape contract matches visualize_model. Put animation controls in
    options: frame_duration and final_hold_duration are milliseconds, and loop is a boolean.
    visualtorch_capabilities reports their defaults and every style-specific option.

    This tool executes source code with the current user's permissions. It is process-separated
    for reliability, not sandboxed for security, so only provide source you trust. The returned
    object describes the GIF written to disk; it does not embed the animation bytes.
    """
    return await _run_cancellable_job(
        run_animation,
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


async def _run_cancellable_job(
    function: Callable[..., dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    """Run a blocking job while propagating MCP cancellation to its worker process."""
    cancel_event = Event()
    call = partial(function, _cancel_event=cancel_event, **kwargs)
    try:
        return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        cancel_event.set()
        raise


@mcp.tool()
def visualtorch_capabilities(style: StyleName | None = None) -> dict[str, object]:
    """Return structured styles, options, palettes, output contracts, and safety constraints.

    Supply a canonical style or alias to limit the styles section, or omit it for the complete
    VisualTorch MCP capability manifest.
    """
    return capabilities_manifest(style)


@mcp.tool()
def visualtorch_reference(style: StyleName | None = None) -> str:
    """Return canonical upstream VisualTorch documentation links for all or one style."""
    return docs_manifest(style)


@mcp.resource("visualtorch://capabilities", mime_type="application/json")
def visualtorch_capabilities_resource() -> str:
    """Return the machine-readable VisualTorch MCP capability manifest as JSON."""
    return json.dumps(capabilities_manifest(), indent=2)


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
    """Return distinct VisualTorch package and MCP API versions."""
    return f"VisualTorch package: {__version__}\nMCP API: {MCP_API_VERSION}"


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser without writing into the MCP stdio stream."""
    parser = argparse.ArgumentParser(
        prog="visualtorch-mcp",
        description="Run the client-neutral VisualTorch MCP server over stdio.",
        epilog="Model source is executed locally; only use source you trust.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (MCP API {MCP_API_VERSION})",
    )
    return parser


def main() -> None:
    """Parse informational flags, then run the stdio MCP server."""
    _parser().parse_args()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
