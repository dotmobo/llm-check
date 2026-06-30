import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from main import (
    MODEL_CONFIG,
    BASE_URL,
    PROMPT_WITH_TOOL,
    TEST_TOOL,
    run_test,
    run_basic_completion_test,
    run_tool_calling_test,
    run_tool_calling_strict_test,
    run_multimodal_test,
    run_reasoning_test,
    run_streaming_test,
    run_transcription_test,
    run_embedding_test,
    run_rerank_test,
    run_model_tests,
    print_report,
    save_json_report,
    _build_report,
    LLMReport,
    ReportOutput,
    EmbeddingResult,
    RerankResult,
    TranscriptionResult,
)


# --- run_basic_completion_test ---


class TestBasicCompletion:
    def test_returns_response_content(self):
        """Basic completion returns the model's response text."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello! How can I help you?", tool_calls=None)
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_basic_completion_test("gpt-oss-120b")

        assert result["response"] == "Hello! How can I help you?"

    def test_handles_empty_content(self):
        """Empty content is stored as empty string."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=None))
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_basic_completion_test("gpt-oss-120b")

        assert result["response"] == ""

    def test_raises_error_on_no_choices(self):
        """Raises ValueError when response has no choices."""
        mock_response = MagicMock()
        mock_response.choices = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No choices returned"):
                run_basic_completion_test("gpt-oss-120b")

    def test_raises_error_on_empty_choices(self):
        """Raises error when choices[0] is None."""
        mock_response = MagicMock()
        mock_response.choices = [None]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(AttributeError):
                run_basic_completion_test("gpt-oss-120b")

    def test_raises_error_on_none_message(self):
        """Raises ValueError when message is None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=None)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No message in response"):
                run_basic_completion_test("gpt-oss-120b")


# --- run_tool_calling_strict_test ---


class TestToolCallingStrict:
    def test_model_calls_tool_strict(self):
        """Model returns a tool call with strict mode."""
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )

        mock_func = MagicMock()
        mock_func.name = "get_current_temperature"
        mock_func.arguments = '{"location":"Paris, France"}'
        mock_tool_call = MagicMock(spec=ChatCompletionMessageToolCall)
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_func
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[mock_tool_call],
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_tool_calling_strict_test("gpt-oss-120b")

        assert result["tool_calls"] is True
        assert result["tool_names"] == ["get_current_temperature"]

    def test_model_does_not_call_tool_strict(self):
        """Model returns no tool calls in strict mode — raises error."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="I'm not sure about the temperature.",
                    tool_calls=None,
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="did not call any tools"):
                run_tool_calling_strict_test("gpt-oss-120b")

    def test_raises_error_on_no_choices_strict(self):
        """Raises ValueError when no choices returned in strict mode."""
        mock_response = MagicMock()
        mock_response.choices = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No choices returned"):
                run_tool_calling_strict_test("gpt-oss-120b")

    def test_raises_error_on_none_message_strict(self):
        """Raises ValueError when message is None in strict mode."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=None)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No message in response"):
                run_tool_calling_strict_test("gpt-oss-120b")


# --- run_tool_calling_test ---


class TestToolCalling:
    def test_model_calls_tool(self):
        """Model returns a tool call when requested."""
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )

        mock_func = MagicMock()
        mock_func.name = "get_current_temperature"
        mock_func.arguments = '{"location":"Paris, France"}'
        mock_tool_call = MagicMock(spec=ChatCompletionMessageToolCall)
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_func
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[mock_tool_call],
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_tool_calling_test("gpt-oss-120b")

        assert result["tool_calls"] is True
        assert result["tool_names"] == ["get_current_temperature"]

    def test_model_does_not_call_tool(self):
        """Model returns no tool calls — raises error."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="I'm not sure about the temperature.",
                    tool_calls=None,
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="did not call any tools"):
                run_tool_calling_test("gpt-oss-120b")

    def test_raises_error_on_no_choices(self):
        """Raises ValueError when no choices returned (server rejects tool call)."""
        mock_response = MagicMock()
        mock_response.choices = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No choices returned"):
                run_tool_calling_test("gpt-oss-120b")

    def test_raises_error_on_none_message(self):
        """Raises ValueError when message is None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=None)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No message in response"):
                run_tool_calling_test("gpt-oss-120b")

    def test_handles_multiple_tool_calls(self):
        """Model can return multiple tool calls."""
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )

        mock_func_1 = MagicMock()
        mock_func_1.name = "get_current_temperature"
        mock_func_1.arguments = '{"location":"Paris, France"}'
        mock_func_2 = MagicMock()
        mock_func_2.name = "get_current_weather"
        mock_func_2.arguments = '{"location":"Lyon, France"}'
        mock_tool_call_1 = MagicMock(spec=ChatCompletionMessageToolCall)
        mock_tool_call_1.type = "function"
        mock_tool_call_1.function = mock_func_1
        mock_tool_call_2 = MagicMock(spec=ChatCompletionMessageToolCall)
        mock_tool_call_2.type = "function"
        mock_tool_call_2.function = mock_func_2

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[mock_tool_call_1, mock_tool_call_2],
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_tool_calling_test("gpt-oss-120b")

        assert result["tool_calls"] is True
        assert result["tool_names"] == [
            "get_current_temperature",
            "get_current_weather",
        ]


# --- run_test ---


class TestRunTest:
    def test_success_case(self):
        """Successful test returns passed=True with result data."""

        def success_func(model):
            return {"data": "ok"}

        result = run_test("test-model", "test", success_func)
        assert result["passed"] is True
        assert result["data"] == "ok"
        assert "latency_ms" in result

    def test_failure_case(self):
        """Failed test returns passed=False with error message."""

        def fail_func(model):
            raise RuntimeError("boom")

        result = run_test("test-model", "test", fail_func)
        assert result["passed"] is False
        assert result["error"] == "boom"
        assert "latency_ms" in result

    def test_latency_is_positive(self):
        """Latency is always positive for real runs."""

        def slow_func(model):
            time.sleep(0.01)
            return {"data": "ok"}

        result = run_test("test-model", "test", slow_func)
        assert result["latency_ms"] > 0


# --- run_model_tests ---


class TestRunModelTests:
    def test_basic_always_runs(self):
        """Basic completion runs regardless of config."""
        with patch(
            "main.run_basic_completion_test", return_value={"response": "ok"}
        ) as mock_basic:
            with patch(
                "main.run_tool_calling_test", return_value={"tool_calls": True}
            ) as mock_tool:
                with patch(
                    "main.run_tool_calling_strict_test",
                    return_value={"tool_calls": True},
                ) as mock_tool_strict:
                    with patch.dict(
                        "main.MODEL_CONFIG",
                        {"test-model": {"tool_calling": False, "model_type": "chat"}},
                    ):
                        result = run_model_tests("test-model")

        assert "basic_completion" in result
        assert "tool_calling" in result
        assert "tool_calling_strict" in result
        assert result["tool_calling"]["skipped"] is True
        assert result["tool_calling_strict"]["skipped"] is True
        assert mock_basic.call_count == 1
        assert mock_tool.call_count == 0
        assert mock_tool_strict.call_count == 0

    def test_all_features_run_when_supported(self):
        """All tests run when all features are supported."""
        with patch(
            "main.run_basic_completion_test", return_value={"response": "ok"}
        ) as mock_basic:
            with patch(
                "main.run_tool_calling_test", return_value={"tool_calls": True}
            ) as mock_tool:
                with patch(
                    "main.run_tool_calling_strict_test",
                    return_value={"tool_calls": True},
                ) as mock_tool_strict:
                    with patch(
                        "main.run_reasoning_test",
                        return_value={"reasoning_content": "3-2=1"},
                    ) as mock_reasoning:
                        with patch(
                            "main.run_multimodal_test",
                            return_value={"response": "logo"},
                        ) as mock_multimodal:
                            with patch(
                                "main.run_streaming_test",
                                return_value={"response": "streamed"},
                            ) as mock_streaming:
                                with patch.dict(
                                    "main.MODEL_CONFIG",
                                    {
                                        "test-model": {
                                            "model_type": "chat",
                                            "tool_calling": True,
                                            "reasoning": True,
                                            "multimodal": True,
                                            "streaming": True,
                                        }
                                    },
                                ):
                                    result = run_model_tests("test-model")

        assert "basic_completion" in result
        assert "tool_calling" in result
        assert "tool_calling_strict" in result
        assert "reasoning" in result
        assert "multimodal" in result
        assert "streaming" in result
        assert result["tool_calling"]["passed"] is True
        assert result["tool_calling_strict"]["passed"] is True
        assert result["reasoning"]["passed"] is True
        assert result["multimodal"]["passed"] is True
        assert result["streaming"]["passed"] is True
        mock_basic.assert_called_once()
        mock_tool.assert_called_once()
        mock_tool_strict.assert_called_once()
        mock_reasoning.assert_called_once()
        mock_multimodal.assert_called_once()
        mock_streaming.assert_called_once()

    def test_skipped_features_not_run(self):
        """Skipped features are not run and show skipped message."""
        with patch("main.run_basic_completion_test", return_value={"response": "ok"}):
            with patch("main.run_tool_calling_test"):
                with patch("main.run_tool_calling_strict_test"):
                    with patch("main.run_reasoning_test"):
                        with patch("main.run_multimodal_test"):
                            with patch("main.run_streaming_test"):
                                with patch.dict(
                                    "main.MODEL_CONFIG",
                                    {
                                        "test-model": {
                                            "model_type": "chat",
                                            "tool_calling": False,
                                            "reasoning": False,
                                            "multimodal": False,
                                            "streaming": False,
                                        }
                                    },
                                ):
                                    result = run_model_tests("test-model")

        assert result["tool_calling"]["skipped"] is True
        assert result["tool_calling_strict"]["skipped"] is True
        assert result["reasoning"]["skipped"] is True
        assert result["multimodal"]["skipped"] is True
        assert result["streaming"]["skipped"] is True


# --- print_report ---


class TestPrintReport:
    def test_prints_pass_fail_report(self, capsys):
        """Report shows pass/fail for each model and test."""
        all_results = {
            "models": {
                "gpt-oss": {
                    "basic_completion": {
                        "passed": True,
                        "latency_ms": 100,
                        "response": "ok",
                    },
                    "tool_calling": {
                        "passed": False,
                        "latency_ms": 50,
                        "error": "nope",
                    },
                },
                "mistral": {
                    "basic_completion": {
                        "passed": True,
                        "latency_ms": 200,
                        "response": "hello",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "gpt-oss" in captured.out
        assert "mistral" in captured.out
        assert "basic_completion" in captured.out
        assert "tool_calling" in captured.out
        assert "1/3 tests failed" in captured.out

    def test_report_all_pass(self, capsys):
        """Summary shows all tests passed when none fail."""
        all_results = {
            "models": {
                "model-a": {
                    "basic_completion": {
                        "passed": True,
                        "latency_ms": 100,
                        "response": "ok",
                    },
                },
                "model-b": {
                    "basic_completion": {
                        "passed": True,
                        "latency_ms": 200,
                        "response": "ok",
                    },
                    "tool_calling": {
                        "passed": True,
                        "latency_ms": 300,
                        "tool_calls": True,
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "All 3 tests passed" in captured.out
        assert "FAIL" not in captured.out

    def test_report_shows_tool_names(self, capsys):
        """Report shows tool names when tools are used."""
        all_results = {
            "models": {
                "model-a": {
                    "tool_calling": {
                        "passed": True,
                        "latency_ms": 100,
                        "tool_calls": True,
                        "tool_names": ["get_weather", "get_time"],
                        "response": "",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "get_weather" in captured.out
        assert "get_time" in captured.out

    def test_report_shows_error(self, capsys):
        """Report shows error messages for failed tests."""
        all_results = {
            "models": {
                "model-a": {
                    "tool_calling": {
                        "passed": False,
                        "latency_ms": 50,
                        "error": "No choices returned",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "No choices returned" in captured.out

    def test_report_shows_reasoning(self, capsys):
        """Report shows reasoning content for reasoning tests."""
        all_results = {
            "models": {
                "model-a": {
                    "reasoning": {
                        "passed": True,
                        "latency_ms": 100,
                        "response": "1",
                        "reasoning_content": "3-2=1",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "PASS" in captured.out
        assert "reasoning" in captured.out
        assert "1" in captured.out

    def test_report_shows_multimodal(self, capsys):
        """Report shows response for multimodal tests."""
        all_results = {
            "models": {
                "model-a": {
                    "multimodal": {
                        "passed": True,
                        "latency_ms": 100,
                        "response": "The image shows a logo",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "logo" in captured.out


# --- Config ---


class TestConfig:
    def test_model_config_initialized(self):
        """MODEL_CONFIG is initialized for all models in MODEL_LIST."""
        assert MODEL_CONFIG
        assert all(isinstance(v, dict) for v in MODEL_CONFIG.values())

    def test_model_config_has_capabilities(self):
        """Every chat model config has expected capability keys."""
        for caps in MODEL_CONFIG.values():
            if caps.get("model_type") == "chat":
                assert "tool_calling" in caps
                assert "reasoning" in caps
                assert "multimodal" in caps
                assert "streaming" in caps

    def test_test_tool_structure(self):
        """TEST_TOOL has the expected structure."""
        assert TEST_TOOL["type"] == "function"
        assert TEST_TOOL["function"]["name"] == "get_current_temperature"
        assert "location" in TEST_TOOL["function"]["parameters"]["properties"]  # ty: ignore[unsupported-operator]

    def test_prompt_with_tool(self):
        """PROMPT_WITH_TOOL mentions tools."""
        assert "tools" in PROMPT_WITH_TOOL.lower()

    def test_base_url_from_env(self):
        """BASE_URL is read from environment."""
        assert BASE_URL.startswith("http")


# --- run_multimodal_test ---


class TestMultimodal:
    def test_multimodal_returns_response(self):
        """Multimodal test returns the model's response text."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="The image shows a logo for ILaaS.",
                )
            )
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_multimodal_test("gpt-oss-120b")

        assert result["response"] == "The image shows a logo for ILaaS."

    def test_multimodal_raises_error_on_no_choices(self):
        """Raises ValueError when response has no choices."""
        mock_response = MagicMock()
        mock_response.choices = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No choices returned"):
                run_multimodal_test("gpt-oss-120b")

    def test_multimodal_raises_error_on_none_message(self):
        """Raises ValueError when message is None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=None)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No message in response"):
                run_multimodal_test("gpt-oss-120b")


# --- run_reasoning_test ---


class TestReasoning:
    def test_reasoning_returns_response_and_reasoning(self):
        """Reasoning test returns both response and reasoning_content."""
        mock_message = MagicMock()
        mock_message.content = "1"
        mock_message.model_extra = {"reasoning_content": "3 - 2 = 1"}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = run_reasoning_test("gpt-oss-120b")

        assert result["response"] == "1"
        assert result["reasoning_content"] == "3 - 2 = 1"

    def test_reasoning_raises_error_on_empty_reasoning(self):
        """Reasoning raises ValueError when no reasoning_content."""
        mock_message = MagicMock()
        mock_message.content = "4"
        mock_message.model_extra = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No reasoning content returned"):
                run_reasoning_test("gpt-oss-120b")

    def test_reasoning_raises_error_on_no_choices(self):
        """Raises ValueError when response has no choices."""
        mock_response = MagicMock()
        mock_response.choices = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No choices returned"):
                run_reasoning_test("gpt-oss-120b")

    def test_reasoning_raises_error_on_none_message(self):
        """Raises ValueError when message is None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=None)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            with pytest.raises(ValueError, match="No message in response"):
                run_reasoning_test("gpt-oss-120b")

    def test_reasoning_passes_extra_body(self):
        """extra_body is passed via kwargs."""
        mock_message = MagicMock()
        mock_message.content = "1"
        mock_message.model_extra = {"reasoning_content": "3 - 2 = 1"}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            run_reasoning_test(
                "gemma", extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )

        MockOpenAI.return_value.chat.completions.create.assert_called_once()
        call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }


# --- Pydantic models ---


class TestPydanticModels:
    def test_report_output_serialization(self):
        """ReportOutput serializes correctly to JSON."""
        report = ReportOutput(
            models={
                "test-model": LLMReport.model_validate(
                    {
                        "basic_completion": {
                            "passed": True,
                            "latency_ms": 100.0,
                            "response": "hello",
                        },
                        "tool_calling": {
                            "passed": True,
                            "latency_ms": 200.0,
                            "tool_calls": True,
                            "tool_names": ["get_temperature"],
                        },
                        "tool_calling_strict": {
                            "passed": True,
                            "latency_ms": 210.0,
                            "tool_calls": True,
                            "tool_names": ["get_temperature"],
                        },
                        "reasoning": {
                            "passed": False,
                            "skipped": True,
                            "latency_ms": 0.0,
                        },
                        "multimodal": {
                            "passed": True,
                            "latency_ms": 300.0,
                            "response": "A cat",
                        },
                        "streaming": {
                            "passed": True,
                            "latency_ms": 150.0,
                            "response": "streamed response",
                        },
                        "transcription": {
                            "passed": True,
                            "latency_ms": 180.0,
                            "response": "transcribed text",
                        },
                        "embedding": {
                            "passed": False,
                            "skipped": True,
                            "latency_ms": 0.0,
                        },
                        "rerank": {
                            "passed": False,
                            "skipped": True,
                            "latency_ms": 0.0,
                        },
                    }
                )
            }
        )
        data = report.model_dump()
        assert "models" in data
        assert "test-model" in data["models"]
        assert data["models"]["test-model"]["basic_completion"]["response"] == "hello"
        assert data["models"]["test-model"]["reasoning"]["skipped"] is True
        assert data["models"]["test-model"]["tool_calling_strict"]["tool_calls"] is True
        assert (
            data["models"]["test-model"]["transcription"]["response"]
            == "transcribed text"
        )

    def test_llm_report_validates(self):
        """LLMReport validates its structure."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 50.0,
                    "response": "hi",
                },
                "tool_calling": {
                    "passed": True,
                    "latency_ms": 60.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "tool_calling_strict": {
                    "passed": True,
                    "latency_ms": 70.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "reasoning": {
                    "passed": True,
                    "latency_ms": 80.0,
                    "response": "1",
                    "reasoning_content": "1+1=2",
                },
                "multimodal": {
                    "passed": True,
                    "latency_ms": 90.0,
                    "response": "Image content",
                },
                "streaming": {
                    "passed": True,
                    "latency_ms": 100.0,
                    "response": "streamed",
                },
                "transcription": {
                    "passed": True,
                    "latency_ms": 110.0,
                    "response": "transcription",
                },
                "embedding": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "rerank": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
            }
        )
        assert report.basic_completion.passed is True
        assert report.tool_calling.tool_calls is True
        assert report.tool_calling_strict.tool_calls is True
        assert report.reasoning.reasoning_content == "1+1=2"
        assert report.streaming.response == "streamed"
        assert report.transcription.response == "transcription"

    def test_skipped_result_defaults(self):
        """Skipped results have correct defaults."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 10.0,
                    "response": "test",
                },
                "tool_calling": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "tool_calling_strict": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "reasoning": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "multimodal": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "streaming": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "transcription": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "embedding": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "rerank": {"passed": False, "skipped": True, "latency_ms": 0.0},
            }
        )
        assert report.tool_calling.skipped is True
        assert report.tool_calling.passed is False
        assert report.tool_calling_strict.skipped is True
        assert report.reasoning.skipped is True
        assert report.multimodal.skipped is True
        assert report.streaming.skipped is True
        assert report.transcription.skipped is True


class TestSaveJsonReport:
    def test_save_creates_file(self, tmp_path):
        """save_json_report creates the output file."""
        test_dir = tmp_path / "test_output"
        test_dir.mkdir()
        report_file = test_dir / "report.json"

        with patch("main._OUTPUT_DIR", test_dir):
            save_json_report(
                {
                    "models": {
                        "m": {
                            "basic_completion": {
                                "passed": True,
                                "latency_ms": 1.0,
                                "response": "ok",
                            },
                            "tool_calling": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0.0,
                            },
                        }
                    }
                }
            )

        assert report_file.exists()
        with open(report_file) as f:
            data = json.load(f)
        assert "models" in data


class TestBuildReport:
    def test_chat_model_strips_embedding_and_rerank(self):
        """Chat model report excludes embedding and rerank keys."""
        with patch(
            "main.MODEL_CONFIG", {"gpt": {"model_type": "chat", "extra_body": None}}
        ):
            result = _build_report(
                {
                    "models": {
                        "gpt": {
                            "basic_completion": {
                                "passed": True,
                                "latency_ms": 10,
                                "response": "hi",
                            },
                            "tool_calling": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "tool_calling_strict": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "reasoning": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "multimodal": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "streaming": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "transcription": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "embedding": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                            "rerank": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                        }
                    }
                }
            )
        model_data = result["models"]["gpt"]
        assert "embedding" not in model_data
        assert "rerank" not in model_data
        assert "basic_completion" in model_data
        assert "tool_calling" in model_data

    def test_embedding_model_has_only_embedding(self):
        """Embedding model report contains only the embedding key."""
        with patch(
            "main.MODEL_CONFIG",
            {"bge-m3": {"model_type": "embedding", "extra_body": None}},
        ):
            result = _build_report(
                {
                    "models": {
                        "bge-m3": {
                            "embedding": {
                                "passed": True,
                                "latency_ms": 42,
                                "embedding_dim": 1024,
                                "embedding_norm": 1.0,
                                "embedding_sample": [0.1, 0.2],
                            },
                            "basic_completion": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                        }
                    }
                }
            )
        model_data = result["models"]["bge-m3"]
        assert set(model_data.keys()) == {"embedding"}

    def test_rerank_model_has_only_rerank(self):
        """Rerank model report contains only the rerank key."""
        with patch(
            "main.MODEL_CONFIG",
            {"bge-reranker": {"model_type": "rerank", "extra_body": None}},
        ):
            result = _build_report(
                {
                    "models": {
                        "bge-reranker": {
                            "rerank": {
                                "passed": True,
                                "latency_ms": 30,
                                "results": [{"index": 0, "relevance_score": 0.9}],
                                "top_score": 0.9,
                                "top_index": 0,
                            },
                            "basic_completion": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0,
                            },
                        }
                    }
                }
            )
        model_data = result["models"]["bge-reranker"]
        assert set(model_data.keys()) == {"rerank"}


# --- run_streaming_test ---


class TestStreaming:
    def _make_chunk(self, content: str | None, idx: int = 0, finish: str = ""):
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.index = idx
        choice.delta = delta
        choice.finish_reason = finish if finish else None
        chunk = MagicMock()
        chunk.choices = [choice]
        return chunk

    def test_streaming_returns_collected_content(self):
        """Streaming test collects all chunks into a single response."""
        stream = [
            self._make_chunk("Hello"),
            self._make_chunk(", "),
            self._make_chunk("world!"),
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = stream
            result = run_streaming_test("gpt-oss-120b")

        assert result["response"] == "Hello, world!"

    def test_streaming_handles_chunks_with_no_content(self):
        """Chunks without content are skipped."""
        stream = [
            self._make_chunk(None),
            self._make_chunk("Hello"),
            self._make_chunk(None),
            self._make_chunk(" world"),
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = stream
            result = run_streaming_test("gpt-oss-120b")

        assert result["response"] == "Hello world"

    def test_streaming_raises_on_empty_response(self):
        """Raises ValueError when stream returns no content."""
        stream = [
            self._make_chunk(None),
            self._make_chunk(None, finish="stop"),
        ]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = stream
            with pytest.raises(ValueError, match="no content"):
                run_streaming_test("gpt-oss-120b")

    def test_streaming_handles_empty_choices(self):
        """Chunks with empty choices list are skipped."""
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        stream = [empty_chunk, self._make_chunk("hello")]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = stream
            result = run_streaming_test("gpt-oss-120b")

        assert result["response"] == "hello"

    def test_streaming_passes_extra_body(self):
        """extra_body is passed via kwargs."""
        stream = [self._make_chunk("ok", finish="stop")]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = stream
            run_streaming_test(
                "gemma", extra_body={"stream_options": {"include_usage": True}}
            )

        MockOpenAI.return_value.chat.completions.create.assert_called_once()
        call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["extra_body"]["stream_options"] == {"include_usage": True}


# --- run_embedding_test ---


class TestEmbedding:
    def test_embedding_returns_dim_norm_and_sample(self):
        """Embedding test returns dimension, norm, and first 5 values."""
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = mock_response
            result = run_embedding_test("bge-m3")

        assert result["embedding_dim"] == 10
        assert result["embedding_norm"] > 0
        assert len(result["embedding_sample"]) == 5
        assert result["embedding_sample"][0] == 0.1

    def test_embedding_raises_on_no_data(self):
        """Raises ValueError when no embedding data is returned."""
        mock_response = MagicMock()
        mock_response.data = None

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = mock_response
            with pytest.raises(ValueError, match="No embedding data returned"):
                run_embedding_test("bge-m3")

    def test_embedding_raises_on_zero_dimension(self):
        """Raises ValueError when embedding dimension is 0."""
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = []
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = mock_response
            with pytest.raises(ValueError, match="Embedding dimension is 0"):
                run_embedding_test("bge-m3")

    def test_embedding_passes_extra_body(self):
        """extra_body is passed via kwargs."""
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = mock_response
            run_embedding_test("bge-m3", extra_body={"encoding_format": "float"})

        MockOpenAI.return_value.embeddings.create.assert_called_once()
        call_kwargs = MockOpenAI.return_value.embeddings.create.call_args[1]
        assert call_kwargs["extra_body"]["encoding_format"] == "float"


# --- Pydantic models for embedding ---


class TestEmbeddingPydantic:
    def test_embedding_result_validation(self):
        """EmbeddingResult validates correctly."""
        result = EmbeddingResult.model_validate(
            {
                "passed": True,
                "latency_ms": 100.0,
                "embedding_dim": 1024,
                "embedding_norm": 1.2345,
                "embedding_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        assert result.passed is True
        assert result.embedding_dim == 1024
        assert result.embedding_norm == 1.2345
        assert result.embedding_sample == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_llm_report_with_embedding(self):
        """LLMReport validates with embedding field."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 50.0,
                    "response": "hi",
                },
                "tool_calling": {
                    "passed": True,
                    "latency_ms": 60.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "tool_calling_strict": {
                    "passed": True,
                    "latency_ms": 70.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "reasoning": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "multimodal": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "streaming": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "embedding": {
                    "passed": True,
                    "latency_ms": 80.0,
                    "embedding_dim": 768,
                    "embedding_norm": 0.9876,
                    "embedding_sample": [0.01, 0.02, 0.03, 0.04, 0.05],
                },
                "rerank": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
            }
        )
        assert report.embedding.passed is True
        assert report.embedding.embedding_dim == 768

    def test_embedding_skipped_result(self):
        """Skipped embedding has correct defaults."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 10.0,
                    "response": "test",
                },
                "tool_calling": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "tool_calling_strict": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "reasoning": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "multimodal": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "streaming": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "embedding": {"passed": False, "skipped": True, "latency_ms": 0.0},
            }
        )
        assert report.embedding.skipped is True
        assert report.embedding.passed is False


# --- run_model_tests embedding routing ---


class TestRunModelTestsEmbedding:
    def test_embedding_model_only_runs_embedding(self):
        """Models with type=embedding only run the embedding test."""
        with patch(
            "main.run_embedding_test",
            return_value={
                "passed": True,
                "embedding_dim": 1024,
                "embedding_norm": 1.0,
                "embedding_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
        ) as mock_embed:
            with patch.dict(
                "main.MODEL_CONFIG",
                {"bge-m3": {"model_type": "embedding"}},
            ):
                result = run_model_tests("bge-m3")

        assert "embedding" in result
        assert result["embedding"]["passed"] is True
        mock_embed.assert_called_once()


class TestRunModelTestsRerank:
    def test_rerank_model_only_runs_rerank(self):
        """Models with type=rerank only run the rerank test."""
        with patch(
            "main.run_rerank_test",
            return_value={
                "passed": True,
                "results": [{"index": 0, "relevance_score": 0.95}],
                "top_score": 0.95,
                "top_index": 0,
            },
        ) as mock_rerank:
            with patch.dict(
                "main.MODEL_CONFIG",
                {"bge-reranker-v2-m3": {"model_type": "rerank"}},
            ):
                result = run_model_tests("bge-reranker-v2-m3")

        assert "rerank" in result
        assert result["rerank"]["passed"] is True
        assert len(result) == 1
        mock_rerank.assert_called_once()


# --- run_rerank_test ---


class TestRerank:
    def test_rerank_returns_scores(self):
        """Rerank test returns ranked results with scores."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.42},
                {"index": 1, "relevance_score": 0.08},
            ],
            "id": "test-id",
        }

        with patch("main.httpx.post", return_value=mock_response):
            result = run_rerank_test("bge-reranker-v2-m3")

        assert result["top_score"] == 0.95
        assert result["top_index"] == 0
        assert len(result["results"]) == 3
        assert result["results"][0]["relevance_score"] == 0.95

    def test_rerank_raises_on_no_results(self):
        """Raises ValueError when no rerank results returned."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        with patch("main.httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="No rerank results returned"):
                run_rerank_test("bge-reranker-v2-m3")

    def test_rerank_raises_on_http_error(self):
        """Raises error when HTTP request fails."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )

        with patch("main.httpx.post", return_value=mock_response):
            with pytest.raises(Exception):
                run_rerank_test("bge-reranker-v2-m3")

    def test_rerank_passes_extra_body(self):
        """extra_body is passed via the payload through httpx."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.95}],
            "id": "test-id",
        }

        with patch("main.httpx.post", return_value=mock_response) as mock_post:
            run_rerank_test(
                "bge-reranker-v2-m3",
                extra_body={"top_n": 5},
            )

        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "bge-reranker-v2-m3"


# --- Pydantic models for rerank ---


class TestRerankPydantic:
    def test_rerank_result_validation(self):
        """RerankResult validates correctly."""
        result = RerankResult.model_validate(
            {
                "passed": True,
                "latency_ms": 100.0,
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.42},
                ],
                "top_score": 0.95,
                "top_index": 0,
            }
        )
        assert result.passed is True
        assert result.top_score == 0.95
        assert result.top_index == 0
        assert len(result.results) == 2

    def test_llm_report_with_rerank(self):
        """LLMReport validates with rerank field."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 50.0,
                    "response": "hi",
                },
                "tool_calling": {
                    "passed": True,
                    "latency_ms": 60.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "tool_calling_strict": {
                    "passed": True,
                    "latency_ms": 70.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "reasoning": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "multimodal": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "streaming": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "embedding": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "rerank": {
                    "passed": True,
                    "latency_ms": 80.0,
                    "results": [{"index": 0, "relevance_score": 0.95}],
                    "top_score": 0.95,
                    "top_index": 0,
                },
            }
        )
        assert report.rerank.passed is True
        assert report.rerank.top_score == 0.95

    def test_rerank_skipped_result(self):
        """Skipped rerank has correct defaults."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 10.0,
                    "response": "test",
                },
                "tool_calling": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "tool_calling_strict": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "reasoning": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "multimodal": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "streaming": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "embedding": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "rerank": {"passed": False, "skipped": True, "latency_ms": 0.0},
            }
        )
        assert report.rerank.skipped is True
        assert report.rerank.passed is False


# --- run_model_tests rerank routing ---


class TestRunModelTestsRerankRouting:
    def test_rerank_model_only_runs_rerank(self):
        """Models with type=rerank only run the rerank test."""
        with patch(
            "main.run_rerank_test",
            return_value={
                "passed": True,
                "results": [{"index": 0, "relevance_score": 0.95}],
                "top_score": 0.95,
                "top_index": 0,
            },
        ) as mock_rerank:
            with patch.dict(
                "main.MODEL_CONFIG",
                {"test-rerank": {"model_type": "rerank"}},
            ):
                result = run_model_tests("test-rerank")

        assert "rerank" in result
        assert result["rerank"]["passed"] is True
        assert len(result) == 1
        mock_rerank.assert_called_once()


# --- run_transcription_test ---


class TestTranscription:
    def test_transcription_returns_response(self):
        """Transcription test returns the model's transcription text."""
        mock_response = MagicMock()
        mock_response.text = "Bonjour, comment allez-vous ?"

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.audio.transcriptions.create.return_value = (
                mock_response
            )
            result = run_transcription_test("whisper-medium")

        assert result["response"] == "Bonjour, comment allez-vous ?"

    def test_transcription_raises_error_on_no_text(self):
        """Raises ValueError when transcription returns no text."""
        mock_response = MagicMock()
        mock_response.text = ""

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.audio.transcriptions.create.return_value = (
                mock_response
            )
            with pytest.raises(ValueError, match="no text"):
                run_transcription_test("whisper-medium")

    def test_transcription_creates_correct_api_call(self):
        """Transcription test uses client.audio.transcriptions.create with the file."""
        mock_response = MagicMock()
        mock_response.text = "transcribed text"

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.audio.transcriptions.create.return_value = (
                mock_response
            )
            run_transcription_test("whisper-medium")

        MockOpenAI.return_value.audio.transcriptions.create.assert_called_once()
        call_kwargs = MockOpenAI.return_value.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-medium"

    def test_transcription_passes_extra_body(self):
        """extra_body is passed via kwargs."""
        mock_response = MagicMock()
        mock_response.text = "transcribed"

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.audio.transcriptions.create.return_value = (
                mock_response
            )
            run_transcription_test(
                "whisper-medium",
                extra_body={"language": "fr", "response_format": "json"},
            )

        MockOpenAI.return_value.audio.transcriptions.create.assert_called_once()
        call_kwargs = MockOpenAI.return_value.audio.transcriptions.create.call_args[1]
        assert call_kwargs["extra_body"]["language"] == "fr"
        assert call_kwargs["extra_body"]["response_format"] == "json"


# --- Pydantic models for transcription ---


class TestTranscriptionPydantic:
    def test_transcription_result_validation(self):
        """TranscriptionResult validates correctly."""
        result = TranscriptionResult.model_validate(
            {
                "passed": True,
                "latency_ms": 100.0,
                "response": "Hello world",
            }
        )
        assert result.passed is True
        assert result.response == "Hello world"
        assert result.latency_ms == 100.0

    def test_llm_report_with_transcription(self):
        """LLMReport validates with transcription field."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 50.0,
                    "response": "hi",
                },
                "tool_calling": {
                    "passed": True,
                    "latency_ms": 60.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "tool_calling_strict": {
                    "passed": True,
                    "latency_ms": 70.0,
                    "tool_calls": True,
                    "tool_names": ["tool1"],
                },
                "reasoning": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "multimodal": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "streaming": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "transcription": {
                    "passed": True,
                    "latency_ms": 80.0,
                    "response": "transcription output",
                },
                "embedding": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "rerank": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
            }
        )
        assert report.transcription.passed is True
        assert report.transcription.response == "transcription output"

    def test_transcription_skipped_result(self):
        """Skipped transcription has correct defaults."""
        report = LLMReport.model_validate(
            {
                "basic_completion": {
                    "passed": True,
                    "latency_ms": 10.0,
                    "response": "test",
                },
                "tool_calling": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "tool_calling_strict": {
                    "passed": False,
                    "skipped": True,
                    "latency_ms": 0.0,
                },
                "reasoning": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "multimodal": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "streaming": {"passed": False, "skipped": True, "latency_ms": 0.0},
                "transcription": {"passed": False, "skipped": True, "latency_ms": 0.0},
            }
        )
        assert report.transcription.skipped is True
        assert report.transcription.passed is False


# --- run_model_tests transcription routing ---


class TestRunModelTestsTranscription:
    def test_transcription_model_only_runs_transcription(self):
        """Models with type=transcription only run the transcription test."""
        with patch(
            "main.run_transcription_test",
            return_value={
                "passed": True,
                "response": "transcription text",
            },
        ) as mock_transcribe:
            with patch.dict(
                "main.MODEL_CONFIG",
                {"whisper-medium": {"model_type": "transcription"}},
            ):
                result = run_model_tests("whisper-medium")

        assert "transcription" in result
        assert result["transcription"]["passed"] is True
        assert len(result) == 1
        mock_transcribe.assert_called_once()


# --- print_report: transcription ---


class TestPrintReportTranscription:
    def test_report_shows_transcription(self, capsys):
        """Report shows response for transcription tests."""
        all_results = {
            "models": {
                "model-a": {
                    "transcription": {
                        "passed": True,
                        "latency_ms": 100,
                        "response": "Bonjour, comment allez-vous ?",
                    },
                },
            }
        }

        print_report(all_results)
        captured = capsys.readouterr()

        assert "PASS" in captured.out
        assert "transcription" in captured.out
        assert "Bonjour" in captured.out
