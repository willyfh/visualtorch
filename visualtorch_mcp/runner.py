"""Subprocess runner for VisualTorch renders."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api_reference import normalize_style_name

DEFAULT_OUTPUT_DIR = Path.cwd() / "visualtorch_outputs"


def normalize_input_shape(value: object) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    """Normalize a JSON-friendly input shape into VisualTorch's tuple format."""
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list | tuple) or not value:
        message = "input_shape must be a non-empty list/tuple, or a JSON string containing one."
        raise ValueError(message)

    has_nested = any(isinstance(item, list | tuple) for item in value)
    has_scalar = any(isinstance(item, int) and not isinstance(item, bool) for item in value)
    if has_nested and has_scalar:
        message = "input_shape must be either one flat shape or a list of per-input shapes, not both."
        raise ValueError(message)

    if has_nested:
        return tuple(_normalize_single_shape(item) for item in value)
    return _normalize_single_shape(value)


def resolve_output_path(output_path: str | None, output_dir: str | None, style: str) -> Path:
    """Resolve or create a deterministic render output path."""
    base_dir = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR
    base_dir = base_dir.resolve()

    if output_path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = base_dir / f"visualtorch_{style}_{stamp}_{uuid.uuid4().hex[:8]}.png"

    if not path.suffix:
        path = path.with_suffix(".png")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def render_model(
    *,
    source: str,
    input_shape: object,
    style: str = "graph",
    model_expression: str = "model",
    output_path: str | None = None,
    output_dir: str | None = None,
    options: dict[str, Any] | None = None,
    workdir: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Render a PyTorch model by invoking the worker process."""
    if not source.strip():
        message = "source must contain Python code that defines the model."
        raise ValueError(message)
    if timeout_seconds < 1:
        message = "timeout_seconds must be at least 1."
        raise ValueError(message)

    canonical_style = normalize_style_name(style)
    normalized_shape = normalize_input_shape(input_shape)
    resolved_output_path = resolve_output_path(output_path, output_dir, canonical_style)
    payload = {
        "source": source,
        "input_shape": normalized_shape,
        "style": canonical_style,
        "model_expression": model_expression,
        "output_path": str(resolved_output_path),
        "options": options or {},
        "workdir": workdir,
    }

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as payload_file:
        json.dump(payload, payload_file)
        payload_file_path = Path(payload_file.name)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "visualtorch_mcp.worker", str(payload_file_path)],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        message = f"VisualTorch render timed out after {timeout_seconds} seconds."
        raise TimeoutError(message) from exc
    finally:
        payload_file_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker failed without output"
        message = f"VisualTorch render failed: {detail}"
        raise RuntimeError(message)

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        message = f"VisualTorch worker returned invalid JSON: {completed.stdout!r}"
        raise RuntimeError(message) from exc

    return result


def _normalize_single_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        message = "each input shape must be a non-empty list/tuple of positive integers."
        raise ValueError(message)

    shape: list[int] = []
    for dimension in value:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            message = "each input shape dimension must be a positive integer."
            raise ValueError(message)
        shape.append(dimension)
    return tuple(shape)
