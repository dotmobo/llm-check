import json
import time
from unittest.mock import MagicMock, patch

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
    run_model_tests,
    print_report,
    save_json_report,
    LLMReport,
    ReportOutput,
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
                        "main.MODEL_CONFIG", {"test-model": {"tool_calling": False}}
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
                            with patch.dict(
                                "main.MODEL_CONFIG",
                                {
                                    "test-model": {
                                        "tool_calling": True,
                                        "reasoning": True,
                                        "multimodal": True,
                                    }
                                },
                            ):
                                result = run_model_tests("test-model")

        assert "basic_completion" in result
        assert "tool_calling" in result
        assert "tool_calling_strict" in result
        assert "reasoning" in result
        assert "multimodal" in result
        assert result["tool_calling"]["passed"] is True
        assert result["tool_calling_strict"]["passed"] is True
        assert result["reasoning"]["passed"] is True
        assert result["multimodal"]["passed"] is True
        mock_basic.assert_called_once()
        mock_tool.assert_called_once()
        mock_tool_strict.assert_called_once()
        mock_reasoning.assert_called_once()
        mock_multimodal.assert_called_once()

    def test_skipped_features_not_run(self):
        """Skipped features are not run and show skipped message."""
        with patch("main.run_basic_completion_test", return_value={"response": "ok"}):
            with patch("main.run_tool_calling_test"):
                with patch("main.run_tool_calling_strict_test"):
                    with patch("main.run_reasoning_test"):
                        with patch("main.run_multimodal_test"):
                            with patch.dict(
                                "main.MODEL_CONFIG",
                                {
                                    "test-model": {
                                        "tool_calling": False,
                                        "reasoning": False,
                                        "multimodal": False,
                                    }
                                },
                            ):
                                result = run_model_tests("test-model")

        assert result["tool_calling"]["skipped"] is True
        assert result["tool_calling_strict"]["skipped"] is True
        assert result["reasoning"]["skipped"] is True
        assert result["multimodal"]["skipped"] is True


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
        """Every model config has expected capability keys."""
        for caps in MODEL_CONFIG.values():
            assert "tool_calling" in caps
            assert "reasoning" in caps
            assert "multimodal" in caps

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

    def test_reasoning_passes_chat_template_kwargs(self):
        """chat_template_kwargs is passed via extra_body."""
        mock_message = MagicMock()
        mock_message.content = "1"
        mock_message.model_extra = {"reasoning_content": "3 - 2 = 1"}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("main.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            run_reasoning_test("gemma", chat_template_kwargs={"enable_thinking": True})

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
            }
        )
        assert report.basic_completion.passed is True
        assert report.tool_calling.tool_calls is True  # ty: ignore[unresolved-attribute]
        assert report.tool_calling_strict.tool_calls is True  # ty: ignore[unresolved-attribute]
        assert report.reasoning.reasoning_content == "1+1=2"  # ty: ignore[unresolved-attribute]

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
            }
        )
        assert report.tool_calling.skipped is True
        assert report.tool_calling.passed is False
        assert report.tool_calling_strict.skipped is True
        assert report.reasoning.skipped is True
        assert report.multimodal.skipped is True

    def test_report_output_invalid_data(self):
        """ReportOutput rejects invalid data."""
        with pytest.raises(ValueError):
            ReportOutput.model_validate(
                {"models": {"bad": {"invalid_key": "value"}}}  # type: ignore[arg-type]
            )


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
                            "tool_calling_strict": {
                                "passed": False,
                                "skipped": True,
                                "latency_ms": 0.0,
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
                        }
                    }
                }
            )

        assert report_file.exists()
        with open(report_file) as f:
            data = json.load(f)
        assert "models" in data
