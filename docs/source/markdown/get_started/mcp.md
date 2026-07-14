# MCP integration

VisualTorch ships an optional, client-neutral [Model Context Protocol
(MCP)](https://modelcontextprotocol.io/) server. Any MCP host that supports local stdio servers can
use it to inspect VisualTorch's capabilities and generate static PNG diagrams or animated GIF
reveals from PyTorch model source. The integration is part of VisualTorch itself; it does not
depend on a client-specific plugin or extension.

## Installation

Install VisualTorch and its MCP dependency in the Python environment that should perform renders:

```bash
python -m pip install "visualtorch[mcp]"
```

For a source checkout, use an editable install:

```bash
python -m pip install -e ".[mcp]"
```

Both commands install the `visualtorch-mcp` console script. Confirm that the same environment is
active when starting it:

```bash
visualtorch-mcp
```

The process communicates only through MCP messages on standard input and standard output. It does
not start an HTTP server. Diagnostic and model-generated output is kept off the protocol stream.
The equivalent module entry point is `python -m visualtorch_mcp`.

## Connect an MCP host

Register `visualtorch-mcp` as a stdio server in your MCP host. Configuration formats vary, but a
typical entry has this shape:

```json
{
  "mcpServers": {
    "visualtorch": {
      "command": "visualtorch-mcp"
    }
  }
}
```

If the console script is not on the host's `PATH`, launch the module with the exact interpreter
where VisualTorch is installed:

```json
{
  "mcpServers": {
    "visualtorch": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "visualtorch_mcp"]
    }
  }
}
```

On Windows, the interpreter commonly ends in `Scripts/python.exe`; on POSIX systems it commonly
ends in `bin/python`.

## Discover capabilities first

Call `visualtorch_capabilities` before constructing a request when the host can do so. It returns a
machine-readable description of:

- the MCP schema version, installed VisualTorch version, stdio transport, and subprocess model;
- accepted `input_shape` forms;
- canonical styles, aliases, and the options supported by each style;
- named palettes;
- static and animation output formats; and
- the available tools and the trusted-code requirement.

Pass an optional style or alias to limit the response to one canonical style. The same data is
available as JSON text from the `visualtorch://capabilities` resource.

## Tools

### `visualize_model`

Render a static PyTorch architecture diagram as a PNG.

### `animate_model`

Render an animated GIF that reveals the architecture one column at a time. It accepts the shared
model and output parameters. Set its animation timing inside `options`:

- `frame_duration`: milliseconds for each intermediate frame (default `600`);
- `final_hold_duration`: milliseconds to hold the completed diagram (default `1500`); and
- `loop`: whether the GIF repeats (default `true`).

### `visualtorch_capabilities`

Return the structured capabilities described above. Its optional `style` argument accepts a
canonical name or compatibility alias.

### `visualtorch_reference`

Return links to the upstream VisualTorch API reference and examples. Its optional `style` argument
filters the links to one style.

## Shared render request schema

`visualize_model` and `animate_model` share these fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `source` | string | yes | Python source that imports its dependencies and defines the model. |
| `input_shape` | array or JSON string | yes | One input shape, or one shape per positional model input; include the batch dimension. |
| `style` | string | no | `graph`, `flow`, or `lenet`; defaults to `graph`. |
| `model_expression` | string | no | Expression evaluated after `source`; defaults to `model`. |
| `output_path` | string | no | Output filename or path. Relative paths are resolved below `output_dir`. |
| `output_dir` | string | no | Base directory for generated output. |
| `options` | object | no | Style-specific VisualTorch options advertised by capabilities. |
| `workdir` | string | no | Existing working directory made available to imports in `source`. |
| `timeout_seconds` | integer | no | Worker timeout in seconds; defaults to `120` and must be between `1` and `600`. |

Use a flat array for one model input:

```json
{"input_shape": [1, 3, 224, 224]}
```

Use a nested array for multiple positional inputs:

```json
{"input_shape": [[1, 3, 224, 224], [1, 10]]}
```

Style aliases are accepted for compatibility: `layered` and `layered_view` select `flow`, while
`lenet_style` and `lenet_view` select `lenet`. Capability responses always identify the canonical
style. Style matching is case-insensitive, and hyphens are normalized to underscores.

### JSON-safe option values

The capabilities manifest describes every common and style-specific option, including its type and
default. Options that are Python objects in the direct VisualTorch API use portable MCP encodings:

- `type_ignore` is a list of Python or `torch.nn` type names, such as `["Dropout", "nn.Flatten"]`;
- `color_map` is an object whose keys use the same type-name format;
- `input_dtype` is a torch dtype name such as `"float32"` or `"torch.long"`, or one dtype/null
  entry per model input; and
- `font` is either `"default"` or `{"path": "/absolute/font.ttf", "size": 16}`.

Color fields accept Pillow-compatible strings or JSON RGB/RGBA arrays such as `[230, 159, 0]`.
The same encoding applies to the `fill` and `outline` values inside each `color_map` override.

Do not pass `to_file` in `options`; use `output_path` and `output_dir`, which the MCP worker owns and
validates.

## Output paths and metadata

If `output_path` is omitted, the server creates a unique filename below `output_dir` (or its
default output directory). The server normalizes static output names to `.png` and animation output
names to `.gif`. It creates missing parent directories, rejects relative paths that escape
`output_dir`, and validates paths before launching the worker.

A successful tool result is structured data containing at least:

| Field | Meaning |
| --- | --- |
| `output_path` | Absolute path to the generated file. |
| `kind` | `image` for a static render or `animation` for a GIF. |
| `style` | Canonical style used for the render. |
| `media_type` | MIME type (`image/png` or `image/gif`). |
| `width`, `height` | Pixel dimensions. |
| `bytes` | Generated file size. |

Static results also report the Pillow `mode` when available. Animation results additionally report
`frame_count`, `durations_ms`, and `loop`, read back from the generated GIF. Hosts should use the
returned metadata instead of inferring format, dimensions, or timing from the request.

## Resources

| URI | Contents |
| --- | --- |
| `visualtorch://capabilities` | Complete machine-readable capabilities as JSON text. |
| `visualtorch://docs` | Links to VisualTorch documentation and examples. |
| `visualtorch://api-reference` | Backward-compatible API reference links. |
| `visualtorch://version` | Installed VisualTorch package version and MCP API schema version. |

## Examples

Static graph request:

```json
{
  "source": "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))",
  "input_shape": [1, 1, 2, 2],
  "style": "graph",
  "output_path": "architecture.png",
  "options": {"show_neurons": false, "palette": "okabe_ito"}
}
```

Animated flow request:

```json
{
  "source": "import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))",
  "input_shape": [1, 1, 2, 2],
  "style": "flow",
  "output_path": "architecture.gif",
  "options": {
    "frame_duration": 300,
    "final_hold_duration": 1000,
    "loop": true
  }
}
```

`model_expression` can name a factory or constructor instead of a pre-built variable:

```json
{
  "source": "import torch\nclass Net(torch.nn.Module):\n    def forward(self, x):\n        return x.relu()",
  "model_expression": "Net()",
  "input_shape": [1, 4]
}
```

## Errors and process isolation

Requests are validated before a worker starts. Invalid styles, shapes, path traversal, working
directories, options, and timeouts produce actionable MCP tool errors. Model import, evaluation,
tracing, and rendering happen in a separate subprocess with a timeout; a failed render therefore
does not terminate the long-running MCP server. Each tool call returns only after its output file
has been written and inspected for metadata.

## Security: trusted source only

The server intentionally executes `source` and evaluates `model_expression` as Python code. The
worker subprocess is a reliability boundary, **not a security sandbox**: supplied code can read or
modify files, access the network, start processes, and use the current user's permissions. Connect
the server only to MCP hosts you trust, review generated tool arguments when the host requests
approval, and render only source you trust. For stronger isolation, run the entire MCP server in a
separately secured operating-system account, container, or virtual machine with limited filesystem
and network access.

VisualTorch deliberately provides this portable stdio MCP surface without bundling configuration,
authentication, branding, or packaging for any particular MCP host.
