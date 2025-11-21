# Contributing

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd compute-eval
uv sync
```

This will create a virtual environment and install all dependencies. To also install development dependencies:

```bash
uv sync --group dev
```

### Environment Setup

Create a `.env` file in the `compute-eval` directory:

```env
NEMO_API_KEY="<PUT-YOUR-KEY-HERE>"
```

or

```env
OPENAI_API_KEY="<PUT-YOUR-KEY-HERE>"
```

if using a custom model with OpenAI API compatibility.

### Linting

You will need to install the Ruff Python formatter and linter. To do this in VSCode is simple, get [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) from the Marketplace
and then add these lines to either your workspace settings.json or your global settings.json

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    },
},
"ruff.organizeImports": true,
```

Everytime you save the files, the linter will automatically lint for you. Depending on your workflow, you might want to have it check and report and then ask for permission to format the files.

## Sharing your contributions

For any additonal contributions that are made, please include a DCO in your commit message: https://wiki.linuxfoundation.org/dco
