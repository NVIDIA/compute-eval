# compute-eval

ComputeEval: Evaluating Large Language Models for CUDA Code Generation

ComputeEval is a framework designed to generate and evaluate CUDA code from Large Language Models.
It features:

- A set of handcrafted CUDA programming challenges ("problem set") designed to evaluate an LLM's capability at writing reliable CUDA code
- Utilities for generating multiple solutions to each challenge ("samples")
- Utilities for functional correctness of generated CUDA code

ComputeEval is currently in Alpha. We plan to refine the evaluation framework
and make frequent updates to the dataset with additional problems spanning all
aspects of CUDA development.

## Setup

### Prerequisites

- Python 3.10+ or above
- NVIDIA GPU with CUDA Toolkit 12 or greater (for evaluation)

### Installation

Install the package:

```bash
# pip
pip install .

# Poetry
poetry install
```

Note: If you use Poetry, version 2.0 or later is recommended.

### API Keys

To query an LLM, you must first obtain an API key from the respective service.

#### NVIDIA NEMO (default)

To use ComputeEval with NVIDIA-hosted models, you need a key from
[build.nvidia.com](https://build.nvidia.com).

1. Go to [build.nvidia.com](https://build.nvidia.com)
1. Sign in with your account
1. Verify that you have sufficient credits to call hosted models
1. Navigate to the desired model and click on it
1. Click on `Get API Key`
1. Copy the generated API key
1. Export it as an environment variable:

```bash
export NEMO_API_KEY="<your-nvidia-key>"
```

#### OpenAI

Follow the instructions in the [OpenAI docs](https://openai.com/index/openai-api),
then:

```bash
export OPENAI_API_KEY="<your-openai-key>"
```

#### Anthropic (Claude)

Follow instruction on [Anthropic docs](https://www.anthropic.com/api), then:

```bash
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

## Usage

**Note:** This repository executes machine-generated CUDA code.
While it's unlikely that the code is malicious, it could still pose potential risks.
Therefore, all code execution requires the `--allow-execution` flag.
We strongly recommend using a sandbox environment (e.g., a Docker container or virtual machine) when running generated code to minimize security risks.

ComputeEval can be configured using a YAML file that defines the parameters to the program.
For example `example_config_gen_samples.yaml`:

```yaml
problem_file: data/cuda_problems_121924.jsonl # Input problems
sample_file: data/samples.jsonl # Generated samples

model: llama-3.1-nemotron-70b-instruct # Model to use
num_samples_per_problem: 3 # Samples to generate per problem
```

Note: Please set NEMO_API_KEY when using a preset NIM model.

- Read the problem_file: `data/cuda_problems_121924.jsonl`
- Generate 3 completions per problem using the `llama-3.1-nemotron-70b-instruct` model
- Write all completions to the output samples file: `data/samples.jsonl`

To use a custom model:

```yaml
problem_file: data/problems.jsonl
sample_file: data/samples.jsonl

num_samples_per_problem: 3

custom_model:
  api_endpoint: https://integrate.api.nvidia.com/v1
  model_id: nvidia/llama-3.1-nemotron-70b-instruct
```

Note: Please set OPENAI_API_KEY when using a custom model.

### Models Available

The models available for completions are listed below:

- "mixtral-8x22b" => mistralai/mixtral-8x22b-instruct-v0.1
- "gemma-2b" => google/gemma-2b
- "llama3.1-8b" => meta/llama-3.1-8b-instruct
- "llama3.1-70b" => meta/llama-3.1-70b-instruct
- "llama3.1-405b" => meta/llama-3.1-405b-instruct
- "llama3.2-1b" => meta/llama-3.2-1b-instruct
- "llama3.2-3b" => meta/llama-3.2-3b-instruct
- "llama3.2-90b" => meta/llama-3.2-90b-vision-instruct
- "llama3.1-nemotron-70b" => nvidia/llama-3.1-nemotron-70b-instruct
- "nemotron-mini-4b" => nvidia/nemotron-mini-4b-instruct
- "starcoder2-7b" => bigcode/starcoder2-7b
- "mistral-nemo-12b" => nv-mistralai/mistral-nemo-12b-instruct
- "openai-" => nv-mistralai/mistral-nemo-12b-instruct

By default, NVIDIA hosted `llama-3.1-70b-instruct` is used.

### Generate samples and evaluate

Generate samples based on the config file:

```bash
compute_eval generate_samples -config_file=example_config_gen_samples.yaml
```

Now you have a `data/samples.jsonl`.

To launch an evaluation on the generated samples create a config file
where the content of `example_config_evalcorrectness.yaml`:

```yaml
sample_file: data/samples.jsonl
problem_file: data/cuda_problems_121924.jsonl

k: [1, 3]
```

```bash
compute_eval evaluate_functional_correctness -config_file=example_config_evalcorrectness.yaml
```

Note: the program will ask you to allow code execution by adding the `--allow-execution` flag.

```bash
compute_eval evaluate_functional_correctness -config_file=example_config_evalcorrectness.yaml --allow-execution
```

- This will read the problems and the sample file
- It will run each of the samples through a functional correctness testing suite
- It will output a `pass@k` dictionary with 2 `pass@k` values for k = 1 nand k = 3

Caveats:

- The `k` argument for `evaluate_functional_correctness` should be a comma-separated e.g., `[1,10]`.
- Note that if you have a list of `k` that you want used in evaluation, then `max(k) <= num_samples_per_problem` else that `k` value will not show up in the pass@k dict generated.

## Command Documentation

### `generate_samples`

This command generates samples for given problems using a specified model and writes them to the specified sample_file.

#### Arguments

- `problem_file` (str): The path to the file containing the problems to generate samples for.
- `sample_file` (str, optional): The path to the file where the generated samples will be written. (default: `generated_samples.jsonl`).
- `num_samples_per_problem` (int, optional): The number of samples to generate per problem (default: 100).
- `n_workers` (int, optional): The number of worker threads to use (default: 20).
- `system_prompt` (str, optional): The system prompt to use (default: a predefined CUDA programming prompt).
- `max_tokens` (int, optional): The maximum number of tokens for the model to generate (default: 1024).
- `print_completions` (bool, optional): Flag to specify if you want the completions printed to stdout. (default: False)
- `model` (str, optional): The model to use for generating samples (default: "llama3.1-70b").
- `model_type` (str, optional): The type of model (default: "instruct").
- `custom_model` (dict, optional): api_endpoint (base url) and model_id (model name) for any model that uses the OpenAI API. Please use the OPENAI_API_KEY to set your credentials when using a custom model.
- `params` (dict, optional): parameters for the chat completions request - temperature, top_p, max_tokens.

### `evaluate_functional_correctness`

This command evaluates the functional correctness of generated samples and outputs a `pass@k` dictionary

#### Arguments

- `sample_file` (str): The path to the file containing the samples to be evaluated.
- `problem_file` (str): The path to the file containing the problems to evaluate against.
- `k` (str, optional): The list of values for k, as a comma-separated string (default: "1,10,100").
- `n_workers` (int, optional): The number of worker threads to use (default: 4).
- `timeout` (float, optional): The timeout for each evaluation in seconds (default: 3.0).
- `save_completions_dir` (str, optional): Directory path where the samples will be stored as .cu files (default: "" i.e not saved)

## Dataset

For more information about the dataset see `DATASET_CARD.md`.

## Contributing

See `contributing.md` for development instructions.
