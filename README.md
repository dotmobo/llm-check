# llm-check

A Python script to verify LLM inference server capabilities (basic completion, tool calling, reasoning, multimodal) for monitoring.

## Features

- Tests basic chat completion
- Tests tool calling
- Tests reasoning content
- Tests multimodal (image understanding)
- Skips tests based on model capabilities defined in config
- Human-readable colored console report
- JSON report output for monitoring tools

## Setup

```bash
uv sync
```

## Configuration

Copy `.config.yaml.example` to `config.yaml` and fill in your credentials:

```yaml
llm:
  base_url: https://your-openai-api.fr/v1
  token: your-api-key

models:
  - name: gpt-oss-120b
    capabilities:
      - tool_calling
      - reasoning
  - name: mistral-small-3.2-24b
    capabilities:
      - tool_calling
      - multimodal
```

Supported capabilities:
- `tool_calling` — the model can call tools/functions
- `reasoning` — the model produces reasoning content
- `multimodal` — the model can understand images

Some models require additional `chat_template_kwargs` for reasoning tests. Add it under the model entry:

```yaml
  - name: gemma
    capabilities:
      - tool_calling
      - reasoning
      - multimodal
    chat_template_kwargs:
      enable_thinking: true
```

## Usage

```bash
uv run python main.py
```

## Development

```bash
uv run ruff check main.py test_main.py   # lint
uv run ruff format main.py test_main.py  # format
uv run ty check                          # type check
uv run pytest test_main.py               # tests
```

## Output

- Console: colored pass/fail report
- `output/report.json`: full JSON report for monitoring tools
