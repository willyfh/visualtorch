# MCP integration

VisualTorch provides an optional [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
server for generating architecture diagrams from model source supplied by an MCP client. The MCP
server uses VisualTorch's public `visualtorch.render(...)` API and writes the generated image to a
local file.

## Installation

Install the optional MCP dependency alongside VisualTorch:

```bash
python -m pip install "visualtorch[mcp]"
```

For a source checkout, install the project in editable mode:

```bash
python -m pip install -e ".[mcp]"
```

Both commands install the `visualtorch-mcp` console script.

## MCP client configuration

Add the server to an MCP client configuration that supports stdio servers:

```json
{
  "mcpServers": {
    "visualtorch": {
      "command": "visualtorch-mcp"
    }
  }
}
```

If the client should use a specific Python environment, configure its interpreter explicitly:

```json
{
  "mcpServers": {
    "visualtorch": {
      "command": "C:/path/to/venv/Scripts/python.exe",
      "args": ["-m", "visualtorch_mcp.server"]
    }
  }
}
```

## Tools and resources

The `visualize_model` tool accepts:

- `source`: Python source that imports its dependencies and defines the model.
- `input_shape`: one input shape including the batch dimension, or one shape per positional input.
- `style`: `graph`, `flow`, or `lenet`. The compatibility aliases `layered`, `layered_view`,
  `lenet_style`, and `lenet_view` are also accepted.
- `model_expression`: an expression evaluated after `source`; it defaults to `model` and can be
  set to an expression such as `Net()` or `build_model()`.
- `output_path`: an optional image path. Relative paths are resolved under `output_dir`.
- `output_dir`: an optional base directory for generated images.
- `options`: VisualTorch render options, such as `{"palette": "dracula", "show_dimension": true}`.
- `workdir`: an optional working directory for the model source.
- `timeout_seconds`: the render subprocess timeout, defaulting to 120 seconds.

The `visualtorch_reference` tool and the `visualtorch://docs` resource expose links to the upstream
VisualTorch API references and examples. The `visualtorch://version` resource reports the MCP
integration version.

## Security

The server intentionally executes the supplied model source and evaluates `model_expression` in a
separate Python subprocess. Only connect it to clients and use source code that you trust. The
subprocess timeout and boundary keep a failed render from crashing the MCP server, but they are not
a security sandbox for hostile Python code.
