"""Worker process that imports VisualTorch and performs one isolated job."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import tempfile
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image, ImageFont

if TYPE_CHECKING:
    from collections.abc import Callable

JobKind = Literal["render", "animate"]


def main() -> None:
    """Execute the JSON job payload passed on the command line."""
    protocol_fd = os.dup(sys.stdout.fileno())
    try:
        # Redirect the OS-level stdout descriptor, not just Python's sys.stdout. This also
        # contains native extensions and subprocesses launched by untrusted model source.
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        payload = _load_payload(sys.argv)
        result = _run_from_payload(payload)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        os.close(protocol_fd)
        raise SystemExit(1) from exc

    with os.fdopen(protocol_fd, "w", encoding="utf-8") as protocol_stdout:
        json.dump(result, protocol_stdout)
        protocol_stdout.write("\n")


def _load_payload(arguments: list[str]) -> dict[str, Any]:
    """Load and validate the worker payload path from command-line arguments."""
    if len(arguments) != 2:
        message = "Expected exactly one JSON payload path."
        raise ValueError(message)
    payload = json.loads(Path(arguments[1]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = "Worker payload must be a JSON object."
        raise TypeError(message)
    return payload


def _run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    job = payload.get("job", "render")
    if job not in {"render", "animate"}:
        message = f"Unsupported worker job {job!r}. Expected 'render' or 'animate'."
        raise ValueError(message)

    workdir = payload.get("workdir")
    if workdir:
        resolved_workdir = Path(workdir).expanduser().resolve()
        if not resolved_workdir.is_dir():
            message = f"workdir is not an existing directory: {resolved_workdir}"
            raise ValueError(message)
        os.chdir(resolved_workdir)
        if str(resolved_workdir) not in sys.path:
            sys.path.insert(0, str(resolved_workdir))

    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        message = "source must be a non-empty string."
        raise ValueError(message)
    namespace = _execute_source(source)

    model_expression = payload.get("model_expression") or "model"
    if not isinstance(model_expression, str):
        message = "model_expression must be a string."
        raise TypeError(message)
    model = eval(model_expression, namespace)  # noqa: S307 - trusted source expression by design.
    if hasattr(model, "eval"):
        model.eval()

    output_path = Path(payload["output_path"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_suffix = ".png" if job == "render" else ".gif"
    if output_path.suffix.lower() != expected_suffix:
        message = f"{job} output_path must end with {expected_suffix}."
        raise ValueError(message)
    if output_path.exists() and output_path.is_dir():
        message = f"output_path points to a directory: {output_path}"
        raise ValueError(message)

    raw_options = payload.get("options") or {}
    if not isinstance(raw_options, dict):
        message = "options must be a JSON object."
        raise TypeError(message)
    options = _coerce_options(namespace, raw_options)
    if job == "animate":
        _validate_animation_options(options)
    staging_path = _resolve_staging_path(payload.get("staging_path"), output_path, expected_suffix)
    try:
        # Render to a fresh, same-directory staging file. This prevents a pre-existing final
        # artifact from being reported as a successful new job and preserves it if rendering fails.
        options["to_file"] = str(staging_path)
        result = _call_visualtorch(
            job=job,
            model=model,
            input_shape=_restore_tuples(payload["input_shape"]),
            style=payload["style"],
            options=options,
        )

        staging_is_empty = not staging_path.is_file() or staging_path.stat().st_size == 0
        if job == "render" and hasattr(result, "save") and staging_is_empty:
            result.save(staging_path, format="PNG")
        if not staging_path.is_file() or staging_path.stat().st_size == 0:
            message = f"VisualTorch {job} completed without creating a non-empty artifact."
            raise RuntimeError(message)

        metadata = _artifact_metadata(staging_path, job, payload["style"])
        staging_path.replace(output_path)
        metadata["output_path"] = str(output_path)
        metadata["bytes"] = output_path.stat().st_size
        return metadata
    finally:
        staging_path.unlink(missing_ok=True)


# Backward-compatible internal name used by existing unit tests and integrations.
def _render_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a legacy payload, defaulting its missing job to a static render."""
    return _run_from_payload(payload)


def _execute_source(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__visualtorch_user_model__"}
    exec(compile(source, "<visualtorch-model>", "exec"), namespace)  # noqa: S102 - intentional model execution.
    return namespace


def _call_visualtorch(
    *,
    job: JobKind,
    model: object,
    input_shape: object,
    style: str,
    options: dict[str, Any],
) -> object:
    visualtorch = importlib.import_module("visualtorch")
    if job == "animate":
        animate = getattr(visualtorch, "animate", None)
        if not callable(animate):
            message = "Installed visualtorch does not expose the animate() API."
            raise RuntimeError(message)
        return animate(model, input_shape, style=style, **options)

    render = getattr(visualtorch, "render", None)
    if callable(render):
        return render(model, input_shape, style=style, **options)

    function = _legacy_render_function(style)
    filtered_options = _filter_kwargs(function, options)
    return function(model, input_shape, **filtered_options)


def _artifact_metadata(output_path: Path, job: JobKind, style: str) -> dict[str, Any]:
    """Read the produced artifact back to verify it and report client-neutral metadata."""
    expected_format = "PNG" if job == "render" else "GIF"
    try:
        with Image.open(output_path) as artifact:
            artifact.verify()
        with Image.open(output_path) as artifact:
            if artifact.format != expected_format:
                message = f"Expected a {expected_format} artifact, received {artifact.format or 'unknown format'}."
                raise RuntimeError(message)
            metadata: dict[str, Any] = {
                "output_path": str(output_path),
                "kind": "image" if job == "render" else "animation",
                "style": style,
                "media_type": "image/png" if job == "render" else "image/gif",
                "width": artifact.width,
                "height": artifact.height,
                "mode": artifact.mode,
                "bytes": output_path.stat().st_size,
            }
            if job == "animate":
                frame_count = getattr(artifact, "n_frames", 1)
                durations: list[int | None] = []
                for frame_index in range(frame_count):
                    artifact.seek(frame_index)
                    durations.append(artifact.info.get("duration"))
                raw_loop = artifact.info.get("loop")
                metadata.update(
                    {
                        "frame_count": frame_count,
                        "durations_ms": durations,
                        "loop": None if raw_loop is None else raw_loop == 0,
                    },
                )
            return metadata
    except (OSError, SyntaxError) as exc:
        message = f"VisualTorch produced an unreadable {expected_format} artifact at {output_path}."
        raise RuntimeError(message) from exc


def _resolve_staging_path(value: object, output_path: Path, suffix: str) -> Path:
    """Validate a parent-owned staging path, or reserve one for legacy direct workers."""
    if value is not None:
        if not isinstance(value, str) or not value.strip():
            message = "staging_path must be a non-empty path string."
            raise ValueError(message)
        candidate = Path(value).expanduser()
        if candidate.is_symlink():
            message = "staging_path must not be a symbolic link."
            raise ValueError(message)
        staging_path = candidate.resolve()
        if staging_path.parent != output_path.parent or staging_path.suffix.lower() != suffix:
            message = "staging_path must be a same-directory file with the expected artifact suffix."
            raise ValueError(message)
        if staging_path == output_path or not staging_path.is_file():
            message = "staging_path must be a reserved file distinct from output_path."
            raise ValueError(message)
        return staging_path

    descriptor, name = tempfile.mkstemp(
        prefix=".visualtorch-mcp-",
        suffix=suffix,
        dir=output_path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _legacy_render_function(style: str) -> Callable[..., object]:
    candidates = {
        "graph": (("visualtorch.graph", "graph_view"),),
        "flow": (("visualtorch.flow", "flow_view"), ("visualtorch.layered", "layered_view")),
        "lenet": (("visualtorch.lenet_style", "lenet_view"),),
    }[style]

    for module_name, function_name in candidates:
        with suppress(ImportError, AttributeError):
            module = importlib.import_module(module_name)
            return getattr(module, function_name)

    message = f"Installed visualtorch does not expose a renderer for style {style!r}."
    raise RuntimeError(message)


def _coerce_options(namespace: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(options)
    if "type_ignore" in coerced and coerced["type_ignore"] is not None:
        if not isinstance(coerced["type_ignore"], list):
            message = "type_ignore must be an array of type-name strings."
            raise TypeError(message)
        coerced["type_ignore"] = [_resolve_type(namespace, item) for item in coerced["type_ignore"]]
    if "color_map" in coerced and coerced["color_map"] is not None:
        if not isinstance(coerced["color_map"], dict):
            message = "color_map must be an object keyed by type-name strings."
            raise TypeError(message)
        coerced["color_map"] = {
            _resolve_type(namespace, key): _coerce_color_map_override(value)
            for key, value in coerced["color_map"].items()
        }
    for name in ("background_fill", "font_color", "connector_fill"):
        if name in coerced:
            coerced[name] = _coerce_color_array(coerced[name], name)
    if "input_dtype" in coerced:
        coerced["input_dtype"] = _coerce_input_dtype(coerced["input_dtype"])
    if "font" in coerced and coerced["font"] is not None:
        coerced["font"] = _coerce_font(coerced["font"])
    return coerced


def _coerce_color_map_override(value: object) -> object:
    """Convert JSON color arrays inside a ``{fill, outline}`` override."""
    if not isinstance(value, dict):
        message = "each color_map override must be an object with fill and/or outline fields."
        raise TypeError(message)
    override = dict(value)
    for name in ("fill", "outline"):
        if name in override:
            override[name] = _coerce_color_array(override[name], f"color_map.{name}")
    return override


def _coerce_color_array(value: object, name: str) -> object:
    """Convert a JSON RGB/RGBA integer array to the tuple expected by Pillow."""
    if not isinstance(value, list):
        return value
    if len(value) not in {3, 4} or any(
        not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel <= 255 for channel in value
    ):
        message = f"{name} color array must contain 3 or 4 integer channels between 0 and 255."
        raise ValueError(message)
    return tuple(value)


def _coerce_input_dtype(value: object) -> object:
    """Convert JSON dtype strings/lists to the torch values VisualTorch expects."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value or any(isinstance(item, list) for item in value):
            message = "input_dtype list must contain one dtype string or null per model input."
            raise ValueError(message)
        return tuple(_coerce_input_dtype(item) for item in value)
    if not isinstance(value, str) or not value.strip():
        message = "input_dtype must be a torch dtype string, null, or a list of those values."
        raise TypeError(message)

    torch = importlib.import_module("torch")
    name = value.strip().removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if dtype is None:
        message = f"Unknown torch input_dtype {value!r}; use a name such as 'float32' or 'torch.long'."
        raise ValueError(message)
    if not isinstance(dtype, torch.dtype):
        message = f"torch.{name} exists but is not a torch dtype."
        raise TypeError(message)
    return dtype


def _coerce_font(value: object) -> object:
    """Load the default font or a JSON ``{path, size}`` descriptor for Pillow."""
    if isinstance(value, str):
        if value != "default":
            message = "font string must be 'default'; custom fonts use {'path': ..., 'size': ...}."
            raise ValueError(message)
        return ImageFont.load_default()
    if isinstance(value, dict):
        unknown = set(value) - {"path", "size"}
        if unknown:
            message = f"font descriptor has unsupported fields: {', '.join(sorted(unknown))}."
            raise ValueError(message)
        path_value = value.get("path")
        size = value.get("size")
    else:
        message = "font must be 'default' or an object with 'path' and 'size'."
        raise TypeError(message)

    if not isinstance(path_value, str) or not path_value.strip():
        message = "font.path must be a non-empty path string."
        raise ValueError(message)
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 512:
        message = "font.size must be an integer between 1 and 512."
        raise ValueError(message)
    font_path = Path(path_value).expanduser().resolve()
    if not font_path.is_file():
        message = f"font file does not exist: {font_path}"
        raise ValueError(message)
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError as exc:
        message = f"font file could not be loaded by Pillow: {font_path}"
        raise ValueError(message) from exc


def _validate_animation_options(options: dict[str, Any]) -> None:
    """Validate the shared JSON animation controls before calling VisualTorch."""
    for name in ("frame_duration", "final_hold_duration"):
        value = options.get(name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            message = f"{name} must be a positive integer number of milliseconds."
            raise ValueError(message)
    if "loop" in options and not isinstance(options["loop"], bool):
        message = "loop must be a boolean."
        raise TypeError(message)


def _resolve_type(namespace: dict[str, Any], value: object) -> type:
    """Resolve a dotted type name without evaluating arbitrary Python expressions."""
    if isinstance(value, type):
        return value
    if not isinstance(value, str):
        message = f"Expected a type name string, got {value!r}."
        raise TypeError(message)

    name = value.strip()
    parts = name.split(".")
    if not name or any(not part.isidentifier() or part.startswith("__") for part in parts):
        message = f"Invalid type name {value!r}; use identifiers such as 'Linear', 'nn.Linear', or 'module.Type'."
        raise ValueError(message)

    if len(parts) == 1:
        if name in namespace:
            resolved = namespace[name]
        else:
            torch_nn = importlib.import_module("torch.nn")
            try:
                resolved = getattr(torch_nn, name)
            except AttributeError as exc:
                message = f"Unknown source or torch.nn type {name!r}."
                raise ValueError(message) from exc
    else:
        root_name, *attributes = parts
        if root_name == "nn":
            resolved = importlib.import_module("torch.nn")
        elif root_name == "torch":
            resolved = importlib.import_module("torch")
        elif root_name in namespace:
            resolved = namespace[root_name]
        else:
            message = f"Unknown type root {root_name!r}; import it in source before referring to {name!r}."
            raise ValueError(message)

        missing = object()
        for attribute in attributes:
            candidate = getattr(resolved, attribute, missing)
            if candidate is missing:
                message = f"Type name {name!r} has no attribute {attribute!r}."
                raise ValueError(message)
            resolved = candidate

    if not isinstance(resolved, type):
        message = f"{value!r} did not resolve to a type."
        raise TypeError(message)
    return resolved


def _filter_kwargs(function: Callable[..., object], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _restore_tuples(value: object) -> object:
    if isinstance(value, list):
        return tuple(_restore_tuples(item) for item in value)
    return value


if __name__ == "__main__":
    main()
