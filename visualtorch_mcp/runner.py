"""Isolated subprocess runner for VisualTorch rendering jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

if TYPE_CHECKING:
    from threading import Event

from .api_reference import normalize_style_name

DEFAULT_OUTPUT_DIR = Path.cwd() / "visualtorch_outputs"
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 600
_ERROR_OUTPUT_LIMIT = 16_000
_WORKER_POLL_INTERVAL_SECONDS = 0.1

JobKind = Literal["render", "animate"]


class JobCancelledError(RuntimeError):
    """Raised internally when an MCP cancellation stops a rendering worker."""


def normalize_input_shape(value: object) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    """Normalize a JSON-friendly input shape into VisualTorch's tuple format."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            message = "input_shape must be valid JSON when provided as a string."
            raise ValueError(message) from exc

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


def resolve_output_path(
    output_path: str | None,
    output_dir: str | None,
    style: str,
    *,
    job: JobKind = "render",
) -> Path:
    """Resolve an output path and normalize its suffix for the requested job."""
    suffix = ".png" if job == "render" else ".gif"
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        message = "output_dir must be a non-empty path string when provided."
        raise ValueError(message)
    base_dir = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR
    base_dir = base_dir.resolve()

    if base_dir.exists() and not base_dir.is_dir():
        message = f"output_dir is not a directory: {base_dir}"
        raise ValueError(message)

    relative_output = False
    if output_path is not None:
        if not isinstance(output_path, str) or not output_path.strip():
            message = "output_path must be a non-empty path string when provided."
            raise ValueError(message)
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            relative_output = True
            path = base_dir / path
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = base_dir / f"visualtorch_{style}_{stamp}_{uuid.uuid4().hex[:8]}{suffix}"

    # VisualTorch/Pillow infer the file format from the suffix. Always force the format
    # promised by the tool, rather than silently writing a single frame for a non-GIF path.
    path = path.with_suffix(suffix).resolve()
    if relative_output and not path.is_relative_to(base_dir):
        message = f"relative output_path must stay within output_dir: {base_dir}"
        raise ValueError(message)
    if path.exists() and path.is_dir():
        message = f"output_path points to a directory: {path}"
        raise ValueError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    _cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Render a PyTorch model to a PNG in an isolated worker process."""
    return _run_job(
        job="render",
        source=source,
        input_shape=input_shape,
        style=style,
        model_expression=model_expression,
        output_path=output_path,
        output_dir=output_dir,
        options=options,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
        cancel_event=_cancel_event,
    )


def animate_model(
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
    _cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Animate a PyTorch model to a GIF in an isolated worker process.

    Animation controls such as ``frame_duration``, ``final_hold_duration``, and
    ``loop`` are supplied through ``options`` together with style-specific options.
    """
    return _run_job(
        job="animate",
        source=source,
        input_shape=input_shape,
        style=style,
        model_expression=model_expression,
        output_path=output_path,
        output_dir=output_dir,
        options=options,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
        cancel_event=_cancel_event,
    )


def _run_job(
    *,
    job: JobKind,
    source: str,
    input_shape: object,
    style: str,
    model_expression: str,
    output_path: str | None,
    output_dir: str | None,
    options: dict[str, Any] | None,
    workdir: str | None,
    timeout_seconds: int,
    cancel_event: Event | None,
) -> dict[str, Any]:
    """Validate and run one render or animation job."""
    source = _require_nonempty_string(source, "source")
    model_expression = _require_nonempty_string(model_expression, "model_expression")
    timeout_seconds = _validate_timeout(timeout_seconds)
    resolved_workdir = _resolve_workdir(workdir)
    if options is not None and not isinstance(options, dict):
        message = "options must be a JSON object."
        raise TypeError(message)

    canonical_style = normalize_style_name(style)
    normalized_shape = normalize_input_shape(input_shape)
    resolved_output_path = resolve_output_path(output_path, output_dir, canonical_style, job=job)
    _raise_if_cancelled(cancel_event)
    staging_path = _reserve_staging_path(resolved_output_path)
    try:
        payload = {
            "job": job,
            "source": source,
            "input_shape": normalized_shape,
            "style": canonical_style,
            "model_expression": model_expression,
            "output_path": str(resolved_output_path),
            "staging_path": str(staging_path),
            "options": options or {},
            "workdir": str(resolved_workdir) if resolved_workdir else None,
        }

        try:
            serialized_payload = json.dumps(payload)
        except (TypeError, ValueError) as exc:
            message = "source, input_shape, and options must contain JSON-serializable values."
            raise TypeError(message) from exc

        completed = _run_serialized_payload(serialized_payload, timeout_seconds, cancel_event)
    finally:
        # The parent owns cleanup so a force-killed or crashed worker cannot leak a partial file.
        staging_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker failed without output"
        detail = detail[-_ERROR_OUTPUT_LIMIT:]
        message = f"VisualTorch {job} failed: {detail}"
        raise RuntimeError(message)

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stdout[-_ERROR_OUTPUT_LIMIT:]
        message = f"VisualTorch worker returned invalid JSON: {detail!r}"
        raise RuntimeError(message) from exc

    if not isinstance(result, dict):
        message = "VisualTorch worker returned a JSON value other than an object."
        raise TypeError(message)
    return result


def _run_serialized_payload(
    serialized_payload: str,
    timeout_seconds: int,
    cancel_event: Event | None,
) -> subprocess.CompletedProcess[str]:
    """Write a temporary payload and remove it after the worker stops."""
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as payload_file:
        payload_file.write(serialized_payload)
        payload_file_path = Path(payload_file.name)

    try:
        return _run_worker(payload_file_path, timeout_seconds, cancel_event)
    finally:
        payload_file_path.unlink(missing_ok=True)


def _run_worker(
    payload_file_path: Path,
    timeout_seconds: int,
    cancel_event: Event | None,
) -> subprocess.CompletedProcess[str]:
    """Run a worker in its own process group so timeout cleanup reaches descendants."""
    _raise_if_cancelled(cancel_event)
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    # A file keeps arbitrary user diagnostics from consuming the MCP host's memory, and unlike
    # a pipe it cannot be held open in a way that blocks us by a user-spawned grandchild.
    with tempfile.TemporaryFile("w+b") as stderr_file:
        process = subprocess.Popen(
            [  # noqa: S603 - fixed interpreter and module invocation.
                sys.executable,
                "-m",
                "visualtorch_mcp.worker",
                str(payload_file_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            env=environment,
            **popen_kwargs,
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _raise_cancelled_worker(process)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _raise_timed_out_worker(process, stderr_file, timeout_seconds)

                try:
                    stdout, _ = process.communicate(
                        timeout=min(_WORKER_POLL_INTERVAL_SECONDS, remaining),
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if process.poll() is None:
                _stop_worker(process)
            raise
        stderr = _read_output_tail(stderr_file)

    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _stop_worker(process: subprocess.Popen[str]) -> None:
    """Terminate a worker tree and reap the protocol process."""
    _terminate_process_tree(process)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _raise_cancelled_worker(process: subprocess.Popen[str]) -> NoReturn:
    _stop_worker(process)
    message = "VisualTorch job was cancelled."
    raise JobCancelledError(message)


def _raise_timed_out_worker(
    process: subprocess.Popen[str],
    stderr_file: Any,  # noqa: ANN401 - binary temporary file protocol.
    timeout_seconds: int,
) -> NoReturn:
    _stop_worker(process)
    stderr = _read_output_tail(stderr_file)
    message = f"VisualTorch job timed out after {timeout_seconds} seconds."
    if stderr.strip():
        message = f"{message} Worker output: {stderr.strip()}"
    raise TimeoutError(message)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort terminate a timed-out worker and any processes it spawned."""
    if os.name == "nt":
        if process.poll() is not None:
            return
        # Python cannot terminate an arbitrary Windows process group. taskkill /T is the
        # platform-supported way to include descendants; fall back to killing the worker.
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                [  # noqa: S603, S607 - fixed Windows process-tree cleanup command.
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        if process.poll() is None:
            process.kill()
        return

    kill_process_group = getattr(os, "killpg", None)
    if kill_process_group is None:
        process.kill()
        return

    try:
        kill_process_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    # The group leader may exit while one of its descendants ignores SIGTERM. Track the
    # process group itself through the grace period, then escalate any surviving group.
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        process.poll()
        try:
            kill_process_group(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    with suppress(ProcessLookupError):
        kill_process_group(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _read_output_tail(stream: Any) -> str:  # noqa: ANN401 - binary temporary file protocol.
    """Read only the bounded tail of worker diagnostics from a binary temporary file."""
    stream.flush()
    stream.seek(0, os.SEEK_END)
    byte_count = stream.tell()
    stream.seek(max(0, byte_count - (_ERROR_OUTPUT_LIMIT * 4)))
    return stream.read().decode("utf-8", errors="replace")[-_ERROR_OUTPUT_LIMIT:]


def _reserve_staging_path(output_path: Path) -> Path:
    """Reserve a same-directory artifact path owned and cleaned by the parent process."""
    descriptor, name = tempfile.mkstemp(
        prefix=".visualtorch-mcp-",
        suffix=output_path.suffix,
        dir=output_path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        message = "VisualTorch job was cancelled."
        raise JobCancelledError(message)


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        message = f"{name} must be a non-empty string."
        raise ValueError(message)
    return value


def _validate_timeout(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        message = "timeout_seconds must be an integer."
        raise TypeError(message)
    if not MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS:
        message = f"timeout_seconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}."
        raise ValueError(message)
    return value


def _resolve_workdir(value: str | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        message = "workdir must be a non-empty path string when provided."
        raise ValueError(message)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        message = f"workdir does not exist: {path}"
        raise ValueError(message)
    if not path.is_dir():
        message = f"workdir is not a directory: {path}"
        raise ValueError(message)
    return path


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
