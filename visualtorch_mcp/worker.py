"""Worker process that imports VisualTorch and renders one image."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> None:
    """Render from a JSON payload path passed on the command line."""
    try:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        result = _render_from_payload(payload)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result))


def _render_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    workdir = payload.get("workdir")
    if workdir:
        os.chdir(workdir)

    namespace = _execute_source(payload["source"])
    model_expression = payload.get("model_expression") or "model"
    model = eval(model_expression, namespace)  # noqa: S307 - trusted local model source is intentional.
    if hasattr(model, "eval"):
        model.eval()

    output_path = Path(payload["output_path"]).expanduser().resolve()
    options = _coerce_options(namespace, payload.get("options") or {})
    options["to_file"] = str(output_path)

    image = _call_visualtorch(
        model=model,
        input_shape=_restore_tuples(payload["input_shape"]),
        style=payload["style"],
        options=options,
    )

    if not output_path.exists() and hasattr(image, "save"):
        image.save(output_path)

    width = height = mode = None
    if hasattr(image, "size"):
        width, height = image.size
        mode = getattr(image, "mode", None)

    return {
        "output_path": str(output_path),
        "style": payload["style"],
        "width": width,
        "height": height,
        "mode": mode,
        "bytes": output_path.stat().st_size if output_path.exists() else None,
    }


def _execute_source(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__visualtorch_user_model__"}
    exec(compile(source, "<visualtorch-model>", "exec"), namespace)  # noqa: S102 - intentional model execution.
    return namespace


def _call_visualtorch(model: object, input_shape: object, style: str, options: dict[str, Any]) -> object:
    visualtorch = importlib.import_module("visualtorch")
    render = getattr(visualtorch, "render", None)
    if callable(render):
        return render(model, input_shape, style=style, **options)

    function = _legacy_render_function(style)
    filtered_options = _filter_kwargs(function, options)
    return function(model, input_shape, **filtered_options)


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
        coerced["type_ignore"] = [_resolve_type(namespace, item) for item in coerced["type_ignore"]]
    if "color_map" in coerced and isinstance(coerced["color_map"], dict):
        coerced["color_map"] = {_resolve_type(namespace, key): value for key, value in coerced["color_map"].items()}
    return coerced


def _resolve_type(namespace: dict[str, Any], value: object) -> type:
    if isinstance(value, type):
        return value
    if not isinstance(value, str):
        message = f"Expected a type name string, got {value!r}."
        raise TypeError(message)

    expression = value if "." in value else f"nn.{value}"
    if "nn" not in namespace:
        with suppress(ImportError):
            namespace["nn"] = importlib.import_module("torch.nn")

    resolved = eval(expression, namespace)  # noqa: S307 - resolves trusted model option types.
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
