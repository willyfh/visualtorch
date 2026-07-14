"""Focused tests for the migrated VisualTorch MCP integration."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import MISSING, fields
from pathlib import Path

import pytest
from visualtorch_mcp.api_reference import capabilities_manifest, normalize_style_name
from visualtorch_mcp.runner import (
    animate_model,
    normalize_input_shape,
    render_model,
    resolve_output_path,
)
from visualtorch_mcp.worker import _coerce_options, _restore_tuples, _validate_animation_options

MODEL_SOURCE = "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("graph", "graph"),
        ("FLOW", "flow"),
        ("layered", "flow"),
        ("layered-view", "flow"),
        (" lenet_style ", "lenet"),
        ("LENET_VIEW", "lenet"),
    ],
)
def test_style_aliases(alias: str, canonical: str) -> None:
    """Normalize supported style aliases to their canonical names."""
    assert normalize_style_name(alias) == canonical


def test_unknown_style_is_rejected() -> None:
    """Reject unsupported style names with an actionable error."""
    with pytest.raises(ValueError, match="Unsupported style"):
        normalize_style_name("not-a-style")


def test_capabilities_discovery_is_structured_and_filterable() -> None:
    """Describe neutral tools, outputs, styles, palettes, and the security boundary."""
    capabilities = capabilities_manifest()

    assert capabilities["server"]["transport"] == "stdio"
    assert capabilities["security"]["sandboxed"] is False
    assert set(capabilities["styles"]) == {"graph", "flow", "lenet"}
    assert capabilities["palettes"]
    assert capabilities["outputs"]["visualize_model"]["media_type"] == "image/png"
    assert capabilities["outputs"]["animate_model"]["media_type"] == "image/gif"
    assert {"visualize_model", "animate_model", "visualtorch_capabilities"} <= set(capabilities["tools"])

    flow = capabilities_manifest("layered")
    assert set(flow["styles"]) == {"flow"}
    assert "layered" in flow["styles"]["flow"]["aliases"]
    assert "frame_duration" in flow["animation_options"]


def test_static_capabilities_match_public_visualtorch_contracts() -> None:
    """Catch drift between the import-free MCP manifest and VisualTorch's public API."""
    visualtorch = importlib.import_module("visualtorch")
    capabilities = capabilities_manifest()
    common_fields = {field.name: field for field in fields(visualtorch.CommonOptions)}
    option_types = {
        "graph": visualtorch.GraphStyleOptions,
        "flow": visualtorch.FlowStyleOptions,
        "lenet": visualtorch.LenetStyleOptions,
    }

    for style, option_type in option_types.items():
        style_fields = {field.name: field for field in fields(option_type)}
        public_fields = {**common_fields, **style_fields}
        advertised = capabilities["styles"][style]["options"]
        assert set(advertised) == set(public_fields)
        for name, field in public_fields.items():
            if field.default is not MISSING:
                assert advertised[name]["default"] == field.default

    assert capabilities["palettes"] == {name: list(colors) for name, colors in sorted(visualtorch.PALETTES.items())}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 3, 8, 8], (1, 3, 8, 8)),
        ("[[1, 3, 8, 8], [1, 5]]", ((1, 3, 8, 8), (1, 5))),
        (((1, 3), (1, 2)), ((1, 3), (1, 2))),
    ],
)
def test_input_shape_normalization(value: object, expected: object) -> None:
    """Normalize flat, JSON, and multi-input shapes into tuples."""
    assert normalize_input_shape(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, [3, 8]],
        [[1, 3], []],
        [[1, 3], [1, 0]],
        [[1, 3], [1, True]],
        "not json",
    ],
)
def test_input_shape_validation(value: object) -> None:
    """Reject malformed shapes before starting a render subprocess."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        normalize_input_shape(value)


def test_output_path_resolution(tmp_path: Path) -> None:
    """Resolve job-specific paths beneath the requested output directory."""
    explicit = resolve_output_path("nested/diagram.jpg", str(tmp_path), "graph")
    assert explicit == (tmp_path / "nested" / "diagram.png").resolve()
    assert explicit.parent.is_dir()

    generated = resolve_output_path(None, str(tmp_path), "flow")
    assert generated.parent == tmp_path.resolve()
    assert generated.name.startswith("visualtorch_flow_")
    assert generated.suffix == ".png"

    animated = resolve_output_path("nested/reveal.png", str(tmp_path), "lenet", job="animate")
    assert animated == (tmp_path / "nested" / "reveal.gif").resolve()

    (tmp_path / "diagram").mkdir()
    normalized_beside_directory = resolve_output_path("diagram", str(tmp_path), "graph")
    assert normalized_beside_directory == (tmp_path / "diagram.png").resolve()


def test_output_path_rejects_relative_escape(tmp_path: Path) -> None:
    """Keep relative artifacts inside the explicitly requested output directory."""
    with pytest.raises(ValueError, match="output_dir"):
        resolve_output_path("../escape.png", str(tmp_path), "graph")


def test_output_path_rejects_file_as_directory(tmp_path: Path) -> None:
    """Reject an output_dir that already points to a regular file."""
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir is not a directory"):
        resolve_output_path(None, str(output_file), "graph")


def test_worker_option_type_coercion_and_tuple_restore() -> None:
    """Coerce JSON-safe type, dtype, and font options for VisualTorch."""
    custom = type("Custom", (), {})
    namespace = {}
    coerced = _coerce_options(
        namespace,
        {
            "type_ignore": ["Linear", "torch.nn.Dropout", custom],
            "color_map": {custom: {"fill": [255, 0, 0], "outline": "black"}},
            "background_fill": [255, 255, 255, 255],
            "input_dtype": ["float32", None],
            "font": "default",
        },
    )
    assert coerced["type_ignore"][0].__name__ == "Linear"
    assert coerced["type_ignore"][1].__name__ == "Dropout"
    assert coerced["type_ignore"][2] is custom
    assert coerced["color_map"] == {custom: {"fill": (255, 0, 0), "outline": "black"}}
    assert coerced["background_fill"] == (255, 255, 255, 255)
    assert str(coerced["input_dtype"][0]) == "torch.float32"
    assert coerced["input_dtype"][1] is None
    assert coerced["font"] is not None
    assert _restore_tuples([[1, [2, 3]], 4]) == ((1, (2, 3)), 4)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"input_dtype": "not-a-dtype"}, ValueError, "Unknown torch input_dtype"),
        ({"font": "comic-sans"}, ValueError, "font string must be 'default'"),
        ({"type_ignore": "Dropout"}, TypeError, "type_ignore must be an array"),
        ({"color_map": ["Linear"]}, TypeError, "color_map must be an object"),
        ({"color_map": {"Linear": "red"}}, TypeError, "color_map override must be an object"),
    ],
)
def test_worker_rejects_invalid_json_option_encodings(
    options: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Reject JSON values that cannot be safely converted to renderer types."""
    with pytest.raises(error, match=match):
        _coerce_options({}, options)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"frame_duration": 0}, ValueError, "frame_duration must be a positive integer"),
        ({"final_hold_duration": True}, ValueError, "final_hold_duration must be a positive integer"),
        ({"loop": 1}, TypeError, "loop must be a boolean"),
    ],
)
def test_worker_rejects_invalid_animation_controls(
    options: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Validate animation controls before calling VisualTorch."""
    with pytest.raises(error, match=match):
        _validate_animation_options(options)


@pytest.mark.parametrize("style", ["graph", "flow", "lenet"])
def test_render_model_worker_smoke(tmp_path: Path, style: str) -> None:
    """Render every canonical style through the isolated worker."""
    options = {"show_input": False}
    if style == "graph":
        options["show_neurons"] = False

    result = render_model(
        source=MODEL_SOURCE,
        input_shape=(1, 1, 2, 2),
        style=style,
        output_path=f"{style}-smoke.png",
        output_dir=str(tmp_path),
        options=options,
        timeout_seconds=30,
    )

    output_path = Path(result["output_path"])
    assert output_path == (tmp_path / f"{style}-smoke.png").resolve()
    assert output_path.is_file()
    assert result["kind"] == "image"
    assert result["style"] == style
    assert result["media_type"] == "image/png"
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["bytes"] == output_path.stat().st_size


@pytest.mark.parametrize("style", ["graph", "flow", "lenet"])
def test_animate_model_worker_smoke(tmp_path: Path, style: str) -> None:
    """Animate every canonical style and report verified GIF metadata."""
    options: dict[str, object] = {
        "show_input": False,
        "frame_duration": 1,
        "final_hold_duration": 1,
        "loop": False,
    }
    if style == "graph":
        options["show_neurons"] = False

    result = animate_model(
        source=MODEL_SOURCE,
        input_shape=(1, 1, 2, 2),
        style=style,
        output_path=f"{style}-reveal.png",
        output_dir=str(tmp_path),
        options=options,
        timeout_seconds=30,
    )

    output_path = Path(result["output_path"])
    assert output_path == (tmp_path / f"{style}-reveal.gif").resolve()
    assert output_path.is_file()
    assert result["kind"] == "animation"
    assert result["style"] == style
    assert result["media_type"] == "image/gif"
    assert result["frame_count"] >= 1
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["bytes"] == output_path.stat().st_size


def test_render_model_keeps_source_stdout_out_of_protocol(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Keep user source output off stdout while returning a parseable result."""
    result = render_model(
        source=(
            "print('source output')\n"
            "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))"
        ),
        input_shape=(1, 1, 2, 2),
        style="graph",
        output_path="source-print.png",
        output_dir=str(tmp_path),
        options={"show_neurons": False},
        timeout_seconds=30,
    )

    captured = capsys.readouterr()
    assert "source output" not in captured.out
    assert Path(result["output_path"]).is_file()


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"source": "   "}, ValueError, "source must be a non-empty string"),
        ({"model_expression": ""}, ValueError, "model_expression must be a non-empty string"),
        ({"options": []}, TypeError, "options must be a JSON object"),
        ({"timeout_seconds": True}, TypeError, "timeout_seconds must be an integer"),
        ({"timeout_seconds": 0}, ValueError, "timeout_seconds must be between"),
        ({"timeout_seconds": 601}, ValueError, "timeout_seconds must be between"),
    ],
)
def test_runner_rejects_invalid_job_arguments(
    tmp_path: Path,
    overrides: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Reject malformed job controls before starting a worker."""
    arguments: dict[str, object] = {
        "source": MODEL_SOURCE,
        "input_shape": [1, 1, 2, 2],
        "output_dir": str(tmp_path),
    }
    arguments.update(overrides)

    with pytest.raises(error, match=match):
        render_model(**arguments)


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_runner_rejects_invalid_workdir(tmp_path: Path, path_kind: str) -> None:
    """Require workdir to name an existing directory."""
    workdir = tmp_path / path_kind
    if path_kind == "file":
        workdir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="workdir"):
        render_model(
            source=MODEL_SOURCE,
            input_shape=[1, 1, 2, 2],
            output_dir=str(tmp_path),
            workdir=str(workdir),
        )


def test_runner_reports_malformed_source(tmp_path: Path) -> None:
    """Surface worker compilation failures as an actionable render error."""
    with pytest.raises(RuntimeError, match=r"(?s)VisualTorch render failed:.*SyntaxError"):
        render_model(
            source="def broken(:\n    pass",
            input_shape=[1, 1, 2, 2],
            output_dir=str(tmp_path),
            timeout_seconds=10,
        )


def test_runner_enforces_timeout(tmp_path: Path) -> None:
    """Stop user source that exceeds the configured worker deadline."""
    with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
        render_model(
            source="import time\ntime.sleep(10)\nimport torch\nmodel = torch.nn.Identity()",
            input_shape=[1, 1, 2, 2],
            output_dir=str(tmp_path),
            timeout_seconds=1,
        )


def test_mcp_cancellation_stops_worker_and_cleans_staging(tmp_path: Path) -> None:
    """Propagate MCP task cancellation into the worker process and parent-owned staging file."""
    pytest.importorskip("mcp")
    visualize_model = importlib.import_module("visualtorch_mcp.server").visualize_model

    marker_path = tmp_path / "forward-started"
    output_path = tmp_path / "cancelled.png"
    source = (
        "import time\n"
        "from pathlib import Path\n"
        "import torch\n"
        f"marker = Path({str(marker_path)!r})\n"
        "class SlowModel(torch.nn.Module):\n"
        "    def forward(self, value):\n"
        "        marker.write_text('started', encoding='utf-8')\n"
        "        time.sleep(30)\n"
        "        return value\n"
        "model = SlowModel()\n"
    )

    async def cancel_running_job() -> None:
        task = asyncio.create_task(
            visualize_model(
                source=source,
                input_shape=[1, 1, 2, 2],
                output_path=str(output_path),
                timeout_seconds=30,
            ),
        )
        for _ in range(200):
            if marker_path.exists():
                break
            await asyncio.sleep(0.05)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pytest.fail("worker did not begin model execution")

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for _ in range(200):
            if not list(tmp_path.glob(".visualtorch-mcp-*")):
                return
            await asyncio.sleep(0.05)
        pytest.fail("cancelled worker did not clean its parent-owned staging file")

    asyncio.run(cancel_running_job())
    assert not output_path.exists()


@pytest.mark.parametrize(("job", "suffix"), [("render", ".png"), ("animate", ".gif")])
def test_worker_protocol_keeps_stdout_json(
    tmp_path: Path,
    job: str,
    suffix: str,
) -> None:
    """Keep static and animation worker stdout parseable despite source output."""
    output_path = tmp_path / f"worker-protocol{suffix}"
    payload_path = tmp_path / "payload.json"
    options: dict[str, object] = {"show_neurons": False, "show_input": False}
    if job == "animate":
        options.update({"frame_duration": 1, "final_hold_duration": 1, "loop": False})
    payload_path.write_text(
        json.dumps(
            {
                "job": job,
                "source": (
                    "print('source output')\n"
                    "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))"
                ),
                "input_shape": [1, 1, 2, 2],
                "style": "graph",
                "model_expression": "model",
                "output_path": str(output_path),
                "options": options,
            },
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "visualtorch_mcp.worker", str(payload_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "source output" not in completed.stdout
    assert "source output" in completed.stderr
    result = json.loads(completed.stdout)
    assert Path(result["output_path"]) == output_path.resolve()
    assert result["kind"] == ("image" if job == "render" else "animation")
    assert output_path.is_file()


def test_render_model_imports_model_from_workdir(tmp_path: Path) -> None:
    """Make the documented workdir available for imports in user source."""
    (tmp_path / "local_model.py").write_text(
        "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))\n",
        encoding="utf-8",
    )

    result = render_model(
        source="from local_model import model",
        input_shape=(1, 1, 2, 2),
        style="graph",
        output_path="workdir-import.png",
        output_dir=str(tmp_path),
        workdir=str(tmp_path),
        options={"show_neurons": False},
        timeout_seconds=30,
    )

    assert Path(result["output_path"]).is_file()


def test_mcp_stdio_end_to_end(tmp_path: Path) -> None:
    """Discover and invoke both artifact paths through the real stdio protocol."""
    mcp_client = pytest.importorskip("mcp")
    stdio_client = importlib.import_module("mcp.client.stdio").stdio_client
    client_session = mcp_client.ClientSession
    server_parameters_type = mcp_client.StdioServerParameters

    executable = shutil.which("visualtorch-mcp")
    if executable is None:
        pytest.fail("mcp is installed, but the visualtorch-mcp console script was not found")

    async def exercise_server() -> tuple[object, object, object, object, object, object, object]:
        # Exercise the module entry point directly. This avoids an extra console-script wrapper
        # process on Windows while speaking the exact same stdio MCP protocol.
        server_parameters = server_parameters_type(command=sys.executable, args=["-m", "visualtorch_mcp"])
        async with (
            stdio_client(server_parameters) as (read_stream, write_stream),
            client_session(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            version = await session.read_resource("visualtorch://version")
            capabilities_resource = await session.read_resource("visualtorch://capabilities")
            capabilities_result = await session.call_tool("visualtorch_capabilities", {"style": "layered"})
            static_result = await session.call_tool(
                "visualize_model",
                {
                    "source": f"print('source output')\n{MODEL_SOURCE}",
                    "input_shape": [1, 1, 2, 2],
                    "output_path": "stdio-static.png",
                    "output_dir": str(tmp_path),
                    "options": {"show_neurons": False, "show_input": False},
                    "timeout_seconds": 30,
                },
            )
            animation_result = await session.call_tool(
                "animate_model",
                {
                    "source": f"print('source output')\n{MODEL_SOURCE}",
                    "input_shape": [1, 1, 2, 2],
                    "style": "flow",
                    "output_path": "stdio-animation.gif",
                    "output_dir": str(tmp_path),
                    "options": {
                        "show_input": False,
                        "frame_duration": 1,
                        "final_hold_duration": 1,
                        "loop": False,
                    },
                    "timeout_seconds": 30,
                },
            )
        return (
            tools,
            resources,
            version,
            capabilities_resource,
            capabilities_result,
            static_result,
            animation_result,
        )

    (
        tools,
        resources,
        version,
        capabilities_resource,
        capabilities_result,
        static_result,
        animation_result,
    ) = asyncio.run(exercise_server())

    assert {"visualize_model", "animate_model", "visualtorch_capabilities"} <= {tool.name for tool in tools.tools}
    assert {"visualtorch://version", "visualtorch://capabilities"} <= {
        str(resource.uri) for resource in resources.resources
    }
    assert version.contents
    capabilities = json.loads(capabilities_resource.contents[0].text)
    assert capabilities["server"]["transport"] == "stdio"
    assert not capabilities_result.isError
    assert not static_result.isError
    assert not animation_result.isError
    assert (tmp_path / "stdio-static.png").is_file()
    assert (tmp_path / "stdio-animation.gif").is_file()
