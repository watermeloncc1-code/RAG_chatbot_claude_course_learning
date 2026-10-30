"""Tests that AIGenerator calls the course search tool correctly."""
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator
from config import config
from search_tools import CourseSearchTool, ToolManager
from conftest import (
    make_response, text_block, thinking_block, tool_use_block,
    text_response, tool_use_response,
)

SEARCH_INPUT = {"query": "what is MCP", "course_name": "MCP"}


@pytest.fixture
def client():
    with patch("ai_generator.anthropic.Anthropic") as anthropic_cls:
        yield anthropic_cls.return_value


@pytest.fixture
def generator(client):
    return AIGenerator(api_key="test-key", model="test-model")


@pytest.fixture
def tool_manager(mock_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_store))
    return manager


@pytest.fixture
def tool_flow(client, generator, tool_manager):
    """Script a tool_use turn followed by a final answer; return the call kwargs."""
    tool_manager.execute_tool = MagicMock(return_value="[MCP - Lesson 1]\nServers expose tools.")
    client.messages.create.side_effect = [
        tool_use_response("search_course_content", SEARCH_INPUT, "toolu_42"),
        text_response("MCP connects apps to tools."),
    ]
    answer = generator.generate_response(
        query="What is MCP?",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )
    first, second = [c.kwargs for c in client.messages.create.call_args_list]
    return answer, first, second, tool_manager


class TestFirstCall:
    def test_passes_tools_with_auto_choice(self, client, generator, tool_manager):
        client.messages.create.return_value = text_response("hi")
        generator.generate_response("q", tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager)
        kwargs = client.messages.create.call_args.kwargs
        assert [t["name"] for t in kwargs["tools"]] == ["search_course_content"]
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "q"}]
        assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    def test_appends_history_to_system_prompt(self, client, generator):
        client.messages.create.return_value = text_response("hi")
        generator.generate_response("q", conversation_history="User: hi\nAssistant: hello")
        system = client.messages.create.call_args.kwargs["system"]
        assert system.startswith(AIGenerator.SYSTEM_PROMPT)
        assert system.endswith("Previous conversation:\nUser: hi\nAssistant: hello")

    def test_no_tool_use_makes_single_call(self, client, generator, tool_manager):
        client.messages.create.return_value = text_response("4")
        answer = generator.generate_response(
            "What is 2+2?", tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager
        )
        assert answer == "4"
        assert client.messages.create.call_count == 1


class TestToolExecution:
    def test_executes_search_tool_with_claude_inputs(self, tool_flow):
        _, _, _, manager = tool_flow
        manager.execute_tool.assert_called_once_with("search_course_content", **SEARCH_INPUT)

    def test_returns_final_answer(self, tool_flow):
        answer, *_ = tool_flow
        assert answer == "MCP connects apps to tools."

    def test_second_call_carries_tool_use_and_result(self, tool_flow):
        _, first, second, _ = tool_flow
        messages = second["messages"]
        assert messages[0] == first["messages"][0]
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0].type == "tool_use"
        assert messages[2] == {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_42",
                "content": "[MCP - Lesson 1]\nServers expose tools.",
            }],
        }
        assert second["system"] == first["system"]

    def test_second_call_defines_tools(self, tool_flow):
        # The API rejects requests containing tool_use/tool_result blocks
        # unless `tools` is defined ("...must define tools").
        _, first, second, _ = tool_flow
        assert second.get("tools") == first["tools"]

    def test_second_call_allows_another_round(self, tool_flow):
        _, _, second, _ = tool_flow
        assert second["tool_choice"] == {"type": "auto"}


def run_scripted(client, generator, tool_manager, responses, query="q"):
    """Script Claude's responses; return the answer and each API call's kwargs."""
    client.messages.create.side_effect = responses
    answer = generator.generate_response(
        query, tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager
    )
    return answer, [c.kwargs for c in client.messages.create.call_args_list]


INPUT_1 = {"query": "lesson 4 topic", "course_name": "MCP", "lesson_number": 4}
INPUT_2 = {"query": "prompt caching"}


@pytest.fixture
def two_round_flow(client, generator, tool_manager):
    """Two search rounds, then the final answer."""
    tool_manager.execute_tool = MagicMock(side_effect=["Lesson 4: Prompt caching", "[Other course]\nCaching."])
    answer, calls = run_scripted(client, generator, tool_manager, [
        tool_use_response("search_course_content", INPUT_1, "toolu_1"),
        tool_use_response("search_course_content", INPUT_2, "toolu_2"),
        text_response("Final."),
    ])
    return answer, calls, tool_manager


class TestSequentialRounds:
    def test_two_rounds_make_three_calls_and_return_final_answer(self, two_round_flow):
        answer, calls, manager = two_round_flow
        assert answer == "Final."
        assert len(calls) == 3
        assert [c.args for c in manager.execute_tool.call_args_list] == [
            ("search_course_content",), ("search_course_content",)
        ]
        assert [c.kwargs for c in manager.execute_tool.call_args_list] == [INPUT_1, INPUT_2]

    def test_only_the_call_after_the_last_round_disables_tools(self, two_round_flow):
        _, calls, _ = two_round_flow
        assert [c["tool_choice"]["type"] for c in calls] == ["auto", "auto", "none"]
        assert all(c["tools"] == calls[0]["tools"] for c in calls)

    def test_final_call_carries_both_rounds(self, two_round_flow):
        _, calls, _ = two_round_flow
        messages = calls[2]["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
        assert messages[0] == {"role": "user", "content": "q"}
        assert messages[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Lesson 4: Prompt caching"}
        ]
        assert messages[4]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "[Other course]\nCaching."}
        ]
        assert calls[1]["messages"] == messages[:3]

    def test_stops_after_max_rounds_even_if_claude_asks_for_more(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(return_value="result")
        answer, calls = run_scripted(client, generator, tool_manager, [
            tool_use_response("search_course_content", INPUT_1, "toolu_1"),
            tool_use_response("search_course_content", INPUT_2, "toolu_2"),
            make_response(
                [text_block("Partial answer."), tool_use_block("search_course_content", INPUT_1, "toolu_3")],
                stop_reason="tool_use",
            ),
        ])
        assert answer == "Partial answer."
        assert len(calls) == 3
        assert tool_manager.execute_tool.call_count == 2

    def test_thinking_blocks_passed_back_unchanged(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(return_value="result")
        first_content = [thinking_block("plan"), tool_use_block("search_course_content", INPUT_1)]
        _, calls = run_scripted(client, generator, tool_manager, [
            make_response(first_content, stop_reason="tool_use"),
            text_response("Done."),
        ])
        assert calls[1]["messages"][1] == {"role": "assistant", "content": first_content}

    def test_parallel_tool_uses_count_as_one_round(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(side_effect=["r1", "r2"])
        _, calls = run_scripted(client, generator, tool_manager, [
            make_response(
                [tool_use_block("search_course_content", INPUT_1, "toolu_a"),
                 tool_use_block("search_course_content", INPUT_2, "toolu_b")],
                stop_reason="tool_use",
            ),
            text_response("Done."),
        ])
        assert calls[1]["messages"][2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "r1"},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": "r2"},
        ]
        assert calls[1]["tool_choice"] == {"type": "auto"}


class TestToolErrors:
    def test_tool_exception_reported_to_claude_and_answer_returned(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(side_effect=RuntimeError("db down"))
        answer, calls = run_scripted(client, generator, tool_manager, [
            tool_use_response("search_course_content", INPUT_1, "toolu_1"),
            text_response("Search is unavailable."),
        ])
        assert answer == "Search is unavailable."
        assert calls[1]["messages"][2]["content"] == [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Tool error: db down",
            "is_error": True,
        }]

    def test_failure_in_first_round_skips_second_round(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(side_effect=RuntimeError("db down"))
        _, calls = run_scripted(client, generator, tool_manager, [
            tool_use_response("search_course_content", INPUT_1, "toolu_1"),
            text_response("Search is unavailable."),
        ])
        assert len(calls) == 2
        assert calls[1]["tool_choice"] == {"type": "none"}

    def test_unknown_tool_result_passed_through_as_normal_result(self, client, generator, tool_manager):
        _, calls = run_scripted(client, generator, tool_manager, [
            tool_use_response("no_such_tool", {}, "toolu_1"),
            text_response("Done."),
        ])
        assert calls[1]["messages"][2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Tool 'no_such_tool' not found"}
        ]
        assert calls[1]["tool_choice"] == {"type": "auto"}


class TestResponseParsing:
    def test_extracts_text_after_thinking_block(self, client, generator):
        # Current models return thinking blocks before the text block.
        client.messages.create.return_value = make_response(
            [thinking_block(), text_block("Answer.")]
        )
        assert generator.generate_response("q") == "Answer."

    def test_extracts_text_when_tool_turn_starts_with_text(self, client, generator, tool_manager):
        tool_manager.execute_tool = MagicMock(return_value="result")
        client.messages.create.side_effect = [
            make_response(
                [text_block("Let me search."), tool_use_block("search_course_content", SEARCH_INPUT)],
                stop_reason="tool_use",
            ),
            make_response([thinking_block(), text_block("Final.")]),
        ]
        answer = generator.generate_response(
            "q", tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager
        )
        assert answer == "Final."


@pytest.mark.live
def test_live_api_accepts_configured_key_and_model(require_api_key):
    """Real call with the .env key and config.ANTHROPIC_MODEL."""
    generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
    answer = generator.generate_response("Reply with the single word: ok")
    assert answer.strip()
