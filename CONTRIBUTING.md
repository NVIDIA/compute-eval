# Contributing

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install with evaluation dependencies:

```bash
git clone <repository-url>
cd compute-eval
uv sync --extra eval
```

To also install development tools (linting, testing):

```bash
uv sync --extra eval --group dev
```

> **NVIDIA-internal only:** there is also an `internal` dependency group
> (`uv sync --group internal`) backing internal-only evaluators. It resolves
> from an NVIDIA-internal package index (configured in `uv.toml`) and is
> automatically excised from public releases; external contributors can ignore it.

### Package Extras

| Extra | Contents |
|-------|----------|
| *(none)* | Core data models, datapack API, HuggingFace loading |
| `eval` | Evaluation harness, CLI, profiling support |
| `gen` | Solution generation harness |
| `all` | Everything |

### Environment Setup

Create a `.env` file in the project directory with your API key:

```env
NEMO_API_KEY="<your-nvidia-key>"
```

or for OpenAI-compatible endpoints:

```env
OPENAI_API_KEY="<your-key>"
```

### Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

Run manually:

```bash
uv run ruff check .
uv run ruff format .
```

For VSCode, install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and add to your settings:

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    }
}
```

Pre-commit hooks run Ruff automatically on every commit:

```bash
uv run pre-commit install
```

## Sharing your contributions

For any additional contributions, please include a DCO in your commit message: https://wiki.linuxfoundation.org/dco
