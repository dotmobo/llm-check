# AGENTS.md

## Project structure

Single-module Python project (`main.py`). Entry point: `python main.py`.
- `test_main.py` — unit tests (pytest, mocked OpenAI responses).
- `config.yaml` — live config (gitignored). Copy from `.config.yaml.example` to set up.
- `data/multimodal_test.png` — test image for multimodal test (base64-encoded at import).
- `output/report.json` — generated JSON report (gitignored).

## Config

`config.yaml` has a `llm` section (`base_url`, `token`) and a `models` section (list of dicts with `name` + `capabilities`).
- Capabilities: `tool_calling`, `reasoning`, `multimodal`.
- Legacy format `model_capabilities` with string codes (`tc`, `r`, `m`) is still supported but deprecated.
- **Never** use `.env` — that was removed. Use `config.yaml` only.
- Models without a capability flag get a `{"passed": false, "skipped": true}` result (not `FAIL`).
- `extra_body` can be added under a model entry for per-model API settings (e.g. `chat_template_kwargs`, `reasoning_effort`).

## Key implementation details

- `main.py` is the only source file. All logic is at module level (config loading, test functions, report, main).
- `MODEL_CONFIG` is a module-level dict built from `config.yaml`'s `models` list.
- `model_extra.get("reasoning_content")` is how reasoning content is accessed (OpenAI-specific).
- `max_tokens=2048` for all tests.
- `_IMAGE_BASE64` is loaded once at import time from `data/multimodal_test.png`.
- Colored console output uses ANSI codes only when `_isatty()` is true.

## Development commands

```bash
uv sync                          # install deps
uv run python main.py            # run the tool
uv run pytest test_main.py       # run tests
uv run ruff check main.py test_main.py   # lint
uv run ruff format main.py test_main.py  # format
uv run ty check                # type check
```

All checks (`ruff check`, `ruff format`, `ty`, `pytest`) must pass before committing.

## Adding a model

Edit `config.yaml` → add a new entry under `models:` with `name` and `capabilities`. No code changes needed.

## Known quirks

- `basic_completion` is always run regardless of capabilities.
- Tool calling, reasoning, multimodal are skipped (not failed) if the model doesn't declare the capability.
- The multimodal test will raise if `data/multimodal_test.png` is missing.
