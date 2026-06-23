# llm-check

A Python script to verify LLM inference server capabilities (basic completion, tool calling, reasoning, multimodal, streaming, embedding, rerank) for monitoring.

## Features

- Tests basic chat completion
- Tests tool calling
- Tests reasoning content
- Tests multimodal (image understanding)
- Tests streaming mode
- Tests embedding (vectorization)
- Tests rerank (document re-ranking)
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
  - name: bge-m3
    type: embedding
  - name: gpt-oss-120b
    capabilities:
      - tool_calling
      - reasoning
  - name: mistral-small-3.2-24b
    capabilities:
      - tool_calling
      - multimodal
  - name: qwen-3.6-35b-instruct
    capabilities:
      - tool_calling
      - multimodal
      - reasoning
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
  - name: gemma-4-31b
    capabilities:
      - tool_calling
      - multimodal
      - reasoning
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
  - name: mistral-small-4-119b
    capabilities:
      - tool_calling
      - multimodal
      - reasoning
    extra_body:
      reasoning_effort: high
```

Each model entry supports a `type` field to specify the model category:

- `chat` (default) — model is tested for chat-based capabilities (`tool_calling`, `reasoning`, `multimodal`, `streaming`)
- `embedding` — model is tested for embedding/vectorization capabilities
- `rerank` — model is tested for document re-ranking capabilities

If `type` is omitted, the model defaults to `chat` behavior.

### Chat model capabilities

- `tool_calling` — the model can call tools/functions
- `reasoning` — the model produces reasoning content
- `multimodal` — the model can understand images
- `streaming` — the model supports streaming responses

### Model type

Some models require `extra_body` for reasoning tests:
- `chat_template_kwargs` — e.g. `enable_thinking: true` (qwen, gemma)
- `reasoning_effort` — e.g. `high` (mistral-small-4)

Add it under the model entry as shown above.

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
