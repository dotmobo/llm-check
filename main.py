import base64
import json
import sys
import time
import httpx
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel, Field

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    """Base result shared by all test types."""

    passed: bool = False
    latency_ms: float = 0.0
    skipped: bool = False
    error: str | None = None


class BasicCompletionResult(TestResult):
    response: str = ""


class ToolCallingResult(TestResult):
    tool_calls: bool = False
    tool_names: list[str] = Field(default_factory=list)


class ReasoningResult(TestResult):
    response: str = ""
    reasoning_content: str = ""


class MultimodalResult(TestResult):
    response: str = ""


class StreamingResult(TestResult):
    response: str = ""


class EmbeddingResult(TestResult):
    embedding_dim: int = 0
    embedding_norm: float = 0.0
    embedding_sample: list[float] = Field(default_factory=list)


class RerankResult(TestResult):
    results: list[dict] = Field(default_factory=list)
    top_score: float = 0.0
    top_index: int = 0


# Sentinel skipped result — immutable copy used everywhere
_SKIPPED_RESULT: dict[str, object] = {
    "passed": False,
    "skipped": True,
    "latency_ms": 0.0,
}


def _skipped() -> dict[str, object]:
    """Return a fresh copy of the skipped result dict."""
    return dict(_SKIPPED_RESULT)


class LLMReport(BaseModel):
    basic_completion: BasicCompletionResult = Field(
        default_factory=BasicCompletionResult
    )
    tool_calling: ToolCallingResult = Field(default_factory=ToolCallingResult)
    tool_calling_strict: ToolCallingResult = Field(default_factory=ToolCallingResult)
    reasoning: ReasoningResult = Field(default_factory=ReasoningResult)
    multimodal: MultimodalResult = Field(default_factory=MultimodalResult)
    streaming: StreamingResult = Field(default_factory=StreamingResult)
    embedding: EmbeddingResult = Field(default_factory=EmbeddingResult)
    rerank: RerankResult = Field(default_factory=RerankResult)


class ReportOutput(BaseModel):
    models: dict[str, LLMReport]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_config = _load_config()
_llm_config = _config.get("llm", {})
BASE_URL: str = _llm_config.get("base_url", "http://localhost:8000/v1")
API_KEY: str = _llm_config.get("token", "")
MODEL_LIST: list = _config.get("models", ["gpt-oss-120b", "mistral-small-3.2-24b"])

MODEL_CONFIG: dict[str, dict[str, Any]] = {}
for _entry in MODEL_LIST:
    if isinstance(_entry, dict):
        _name = _entry.get("name", "").strip()
        if not _name:
            print(
                f"[WARN] Skipping model entry with empty name: {_entry}",
                file=sys.stderr,
            )
            continue
        _model_type = _entry.get("type", "chat")
        _caps: list = _entry.get("capabilities", [])
        if _model_type == "chat":
            MODEL_CONFIG[_name] = {
                "model_type": "chat",
                "tool_calling": "tool_calling" in _caps,
                "reasoning": "reasoning" in _caps,
                "multimodal": "multimodal" in _caps,
                "streaming": "streaming" in _caps,
                "extra_body": _entry.get("extra_body"),
            }
        else:
            MODEL_CONFIG[_name] = {
                "model_type": _model_type,
                "extra_body": _entry.get("extra_body"),
            }
    elif isinstance(_entry, str) and _entry.strip():
        MODEL_CONFIG[_entry] = {
            "model_type": "chat",
            "tool_calling": True,
            "reasoning": False,
            "multimodal": False,
            "streaming": True,
            "extra_body": None,
        }
    else:
        print(f"[WARN] Ignoring invalid model entry: {_entry!r}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Multimodal test image
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_IMAGE_PATH = _DATA_DIR / "multimodal_test.png"
_IMAGE_BASE64: str = ""
if _IMAGE_PATH.exists():
    _IMAGE_BASE64 = base64.b64encode(_IMAGE_PATH.read_bytes()).decode("utf-8")

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

TEST_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}

TEST_TOOL_STRICT: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT_WITH_TOOL = (
    "What is the current temperature in Paris, France? "
    "Use the available tools to answer."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    """High-resolution wall-clock time in milliseconds."""
    return time.perf_counter() * 1000


def run_test(model: str, test_name: str, func, **kwargs) -> dict[str, Any]:
    """Run *func(model, **kwargs)* and wrap result with pass/fail + latency."""
    start = _now_ms()
    try:
        result = func(model, **kwargs)
        return {
            "passed": True,
            "latency_ms": round(_now_ms() - start, 1),
            **result,
        }
    except Exception as e:
        print(f"  [WARN] {model}/{test_name}: {e}", file=sys.stderr)
        return {
            "passed": False,
            "latency_ms": round(_now_ms() - start, 1),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def run_basic_completion_test(
    model: str, extra_body: dict | None = None
) -> dict[str, Any]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'hello' in exactly two words."}],
        **kwargs,
    )
    if not response.choices:
        raise ValueError("No choices returned")
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError("No message in response")
    return {"response": choice.message.content or ""}


def run_tool_calling_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_WITH_TOOL}],
        tools=[TEST_TOOL],
        **kwargs,
    )
    if not response.choices:
        raise ValueError(
            "No choices returned — model likely does not support tool calling"
        )
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError(
            "No message in response — model likely does not support tool calling"
        )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        raise ValueError("Model did not call any tools")
    tool_names = [
        tc.function.name
        for tc in tool_calls
        if isinstance(tc, ChatCompletionMessageToolCall)
    ]
    return {"tool_calls": True, "tool_names": tool_names}


def run_tool_calling_strict_test(
    model: str, extra_body: dict | None = None
) -> dict[str, Any]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_WITH_TOOL}],
        tools=[TEST_TOOL_STRICT],
        **kwargs,
    )
    if not response.choices:
        raise ValueError(
            "No choices returned — model likely does not support tool calling"
        )
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError(
            "No message in response — model likely does not support tool calling"
        )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        raise ValueError("Model did not call any tools")
    tool_names = [
        tc.function.name
        for tc in tool_calls
        if isinstance(tc, ChatCompletionMessageToolCall)
    ]
    return {"tool_calls": True, "tool_names": tool_names}


def run_multimodal_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    if not _IMAGE_BASE64:
        raise ValueError(
            "No test image found — place data/multimodal_test.png next to the script"
        )
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image? Answer in one sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_IMAGE_BASE64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        **kwargs,
    )
    if not response.choices:
        raise ValueError(
            "No choices returned — model likely does not support multimodal"
        )
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError(
            "No message in response — model likely does not support multimodal"
        )
    return {"response": choice.message.content or ""}


def run_reasoning_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "If I have 3 apples and give 2 to a friend, "
                    "how many do I have left? Answer with just the number."
                ),
            }
        ],
        **kwargs,
    )
    if not response.choices:
        raise ValueError(
            "No choices returned — model likely does not support reasoning"
        )
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError(
            "No message in response — model likely does not support reasoning"
        )
    extra = choice.message.model_extra or {}
    # Fallback to "reasoning" for older models that don't use "reasoning_content" key
    reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not reasoning:
        raise ValueError("No reasoning content returned")
    return {"response": choice.message.content or "", "reasoning_content": reasoning}


def run_streaming_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kwargs: dict[str, Any] = {"max_tokens": 2048, "stream": True}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 5, one number per line.",
            }
        ],
        **kwargs,
    )
    parts: list[str] = []
    for chunk in response:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta is None:
            continue
        if choice.delta.content is not None:
            parts.append(choice.delta.content)
    full_response = "".join(parts)
    if not full_response.strip():
        raise ValueError("Streaming returned no content")
    return {"response": full_response}


def run_embedding_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=httpx.Timeout(timeout=30.0, connect=10.0),
    )
    kwargs: dict[str, Any] = {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.embeddings.create(
        model=model,
        input="What is machine learning?",
        **kwargs,
    )
    if not response.data:
        raise ValueError("No embedding data returned")
    embedding = response.data[0].embedding
    dim = len(embedding)
    if dim == 0:
        raise ValueError("Embedding dimension is 0")
    norm = sum(v**2 for v in embedding) ** 0.5
    sample = embedding[:5]
    return {
        "embedding_dim": dim,
        "embedding_norm": round(norm, 4),
        "embedding_sample": [round(v, 6) for v in sample],
    }


def run_rerank_test(model: str, extra_body: dict | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    payload = {
        "model": model,
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence that focuses on "
            "systems that learn from data.",
            "The capital of France is Paris.",
            "Quantum computing uses quantum mechanical phenomena to perform computations.",
        ],
        "top_n": 3,
    }
    resp = httpx.post(
        f"{BASE_URL}/rerank",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError("No rerank results returned")
    top = results[0]
    top_index = top.get("index", 0)
    top_score = top.get("relevance_score", 0.0)
    normalized = [
        {"index": r["index"], "relevance_score": round(r["relevance_score"], 6)}
        for r in results
    ]
    return {
        "results": normalized,
        "top_score": round(top_score, 6),
        "top_index": top_index,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_model_tests(model: str) -> dict[str, Any]:
    config = MODEL_CONFIG.get(model, {})
    results: dict[str, Any] = {}
    _extra_body = config.get("extra_body") or {}
    _model_type = config.get("model_type", "chat")

    if _model_type == "embedding":
        return {
            "embedding": run_test(
                model,
                "embedding",
                run_embedding_test,
                extra_body=_extra_body,
            )
        }

    if _model_type == "rerank":
        return {
            "rerank": run_test(
                model,
                "rerank",
                run_rerank_test,
                extra_body=_extra_body,
            )
        }

    results["basic_completion"] = run_test(
        model,
        "basic_completion",
        run_basic_completion_test,
        extra_body=_extra_body,
    )

    results["tool_calling"] = (
        run_test(
            model,
            "tool_calling",
            run_tool_calling_test,
            extra_body=_extra_body,
        )
        if config.get("tool_calling")
        else _skipped()
    )
    results["tool_calling_strict"] = (
        run_test(
            model,
            "tool_calling_strict",
            run_tool_calling_strict_test,
            extra_body=_extra_body,
        )
        if config.get("tool_calling")
        else _skipped()
    )
    results["reasoning"] = (
        run_test(
            model,
            "reasoning",
            run_reasoning_test,
            extra_body=_extra_body,
        )
        if config.get("reasoning")
        else _skipped()
    )
    results["multimodal"] = (
        run_test(
            model,
            "multimodal",
            run_multimodal_test,
            extra_body=_extra_body,
        )
        if config.get("multimodal")
        else _skipped()
    )
    results["streaming"] = (
        run_test(
            model,
            "streaming",
            run_streaming_test,
            extra_body=_extra_body,
        )
        if config.get("streaming")
        else _skipped()
    )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(__file__).parent / "output"
_console = Console()


def print_report(all_results: dict[str, Any]) -> None:
    total = passed = failed = skipped = 0

    for model_name, tests in all_results["models"].items():
        table = Table(
            title=model_name,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold cyan",
        )
        table.add_column("Test", style="dim", min_width=20)
        table.add_column("Status", justify="center", min_width=6)
        table.add_column("Latency", justify="right", min_width=8)
        table.add_column("Detail", no_wrap=False)

        for test_name, result in tests.items():
            total += 1
            is_skipped = result.get("skipped") is True

            if is_skipped:
                skipped += 1
                status = Text("SKIP", style="yellow")
            elif result["passed"]:
                passed += 1
                status = Text("PASS", style="green")
            else:
                failed += 1
                status = Text("FAIL", style="bold red")

            latency = f"{result.get('latency_ms', 0):.0f} ms"

            detail = Text()
            if not is_skipped:
                if "error" in result:
                    detail = Text(result["error"], style="red")
                elif result.get("tool_names"):
                    detail = Text(
                        f"tools: {', '.join(result['tool_names'])}", style="dim"
                    )
                elif result.get("reasoning_content"):
                    resp = result["reasoning_content"][:120]
                    suffix = "…" if len(result["reasoning_content"]) > 120 else ""
                    detail = Text(f"{resp}{suffix}", style="dim")
                elif result.get("embedding_dim"):
                    sample_str = ", ".join(
                        f"{v:.4f}" for v in result.get("embedding_sample", [])
                    )
                    detail = Text(
                        f"dim={result['embedding_dim']}, norm={result.get('embedding_norm', 0):.4f}, sample=[{sample_str}]",
                        style="dim",
                    )
                elif result.get("results"):
                    scores_str = ", ".join(
                        f"{r['relevance_score']:.4f}" for r in result.get("results", [])
                    )
                    detail = Text(
                        f"top_score={result.get('top_score', 0):.4f}, top_idx={result.get('top_index', 0)}, scores=[{scores_str}]",
                        style="dim",
                    )
                elif result.get("response"):
                    resp = result["response"][:120]
                    suffix = "…" if len(result["response"]) > 120 else ""
                    detail = Text(f"{resp}{suffix}", style="dim")

            table.add_row(test_name, status, latency, detail)

        _console.print()
        _console.print(table)

    # Summary panel
    _console.print()
    if failed:
        _console.print(f"  [bold red]✗ {failed}/{total} tests failed[/]")
    elif skipped:
        testable = total - skipped
        _console.print(
            f"  [green]{passed}/{testable} tests passed[/]  [yellow]{skipped} skipped[/]"
        )
    else:
        _console.print(f"  [bold green]✓ All {total} tests passed.[/]")
    _console.print()


_RESULT_TYPES: dict[str, type[BaseModel]] = {
    "basic_completion": BasicCompletionResult,
    "tool_calling": ToolCallingResult,
    "tool_calling_strict": ToolCallingResult,
    "reasoning": ReasoningResult,
    "multimodal": MultimodalResult,
    "streaming": StreamingResult,
    "embedding": EmbeddingResult,
    "rerank": RerankResult,
}


def _model_type_keys(model_type: str) -> set[str]:
    """Return the set of test keys to keep for a given model type."""
    if model_type == "embedding":
        return {"embedding"}
    if model_type == "rerank":
        return {"rerank"}
    return set(_RESULT_TYPES) - {"embedding", "rerank"}


def _build_report(all_results: dict[str, Any]) -> dict[str, Any]:
    """Build filtered report dict by model type — no Pydantic wrapper needed."""
    models = {}
    for model_name, tests in all_results.get("models", {}).items():
        config = MODEL_CONFIG.get(model_name, {})
        _model_type = config.get("model_type", "chat")
        allowed = _model_type_keys(_model_type)
        entry: dict[str, Any] = {}
        for key in allowed:
            if key in tests:
                entry[key] = _RESULT_TYPES[key].model_validate(tests[key]).model_dump()
        models[model_name] = entry
    return {"models": models}


def save_json_report(all_results: dict[str, Any]) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _OUTPUT_DIR / "report.json"
    report = _build_report(all_results)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _console.print(f"\n[dim]JSON report written to[/] [cyan]{report_path}[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not MODEL_CONFIG:
        _console.print("[bold red][ERROR][/] No models configured — check config.yaml")
        sys.exit(1)

    all_results: dict[str, Any] = {"models": {}}

    for model in MODEL_CONFIG:
        _console.print(f"[dim]Testing model:[/] [cyan]{model}[/] …")
        all_results["models"][model] = run_model_tests(model)

    print_report(all_results)
    save_json_report(all_results)


if __name__ == "__main__":
    main()
