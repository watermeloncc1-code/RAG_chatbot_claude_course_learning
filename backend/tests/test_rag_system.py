"""Tests for how RAGSystem handles content-related questions."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from conftest import MCP_COURSE, text_response, tool_use_response

from config import config
from rag_system import RAGSystem


@pytest.fixture
def rag_config(chroma_copy):
    return replace(config, CHROMA_PATH=chroma_copy, ANTHROPIC_API_KEY="test-key")


@pytest.fixture
def rag(rag_config):
    with patch("ai_generator.anthropic.Anthropic") as anthropic_cls:
        system = RAGSystem(rag_config)
        system.client = anthropic_cls.return_value
        yield system


@pytest.fixture
def content_answer(rag):
    """Run a content question where Claude searches the MCP course, lesson 1."""
    rag.client.messages.create.side_effect = [
        tool_use_response(
            "search_course_content",
            {"query": "what is covered", "course_name": "MCP", "lesson_number": 1},
        ),
        text_response("Lesson 1 introduces MCP."),
    ]
    session_id = rag.session_manager.create_session()
    answer, sources = rag.query(
        "What is covered in lesson 1 of the MCP course?", session_id
    )
    return answer, sources, session_id


class TestContentQuery:
    def test_returns_final_answer(self, content_answer):
        answer, _, _ = content_answer
        assert answer == "Lesson 1 introduces MCP."

    def test_prompt_wraps_question_and_offers_search_tool(self, rag, content_answer):
        first = rag.client.messages.create.call_args_list[0].kwargs
        assert first["messages"][0]["content"] == (
            "Answer this question about course materials: "
            "What is covered in lesson 1 of the MCP course?"
        )
        assert [t["name"] for t in first["tools"]] == ["search_course_content"]

    def test_real_search_results_are_sent_to_claude(self, rag, content_answer):
        second = rag.client.messages.create.call_args_list[1].kwargs
        tool_result = second["messages"][-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["content"].startswith(f"[{MCP_COURSE} - Lesson 1]")

    def test_returns_sources_with_links(self, content_answer):
        _, sources, _ = content_answer
        assert sources
        assert sources[0]["label"] == f"{MCP_COURSE} - Lesson 1"
        assert sources[0]["link"]

    def test_sources_reset_after_query(self, rag, content_answer):
        assert rag.tool_manager.get_last_sources() == []

    def test_session_records_exchange(self, rag, content_answer):
        _, _, session_id = content_answer
        history = rag.session_manager.get_conversation_history(session_id)
        assert "User: What is covered in lesson 1 of the MCP course?" in history
        assert "Assistant: Lesson 1 introduces MCP." in history


def test_two_searches_return_sources_from_both(rag):
    rag.client.messages.create.side_effect = [
        tool_use_response(
            "search_course_content",
            {"query": "intro", "course_name": "MCP", "lesson_number": 1},
            "toolu_1",
        ),
        tool_use_response(
            "search_course_content",
            {"query": "intro", "course_name": "MCP", "lesson_number": 2},
            "toolu_2",
        ),
        text_response("Lessons 1 and 2 compared."),
    ]
    answer, sources = rag.query("Compare lessons 1 and 2 of the MCP course")
    assert answer == "Lessons 1 and 2 compared."
    labels = [s["label"] for s in sources]
    assert f"{MCP_COURSE} - Lesson 1" in labels
    assert f"{MCP_COURSE} - Lesson 2" in labels
    assert rag.client.messages.create.call_count == 3


def test_general_question_returns_no_sources(rag):
    rag.client.messages.create.return_value = text_response("4")
    answer, sources = rag.query("What is 2+2?")
    assert answer == "4"
    assert sources == []
    assert rag.client.messages.create.call_count == 1


@pytest.mark.live
def test_live_content_query_end_to_end(chroma_copy, require_api_key):
    """The same path the UI uses, with the real key, model and Chroma index."""
    rag = RAGSystem(replace(config, CHROMA_PATH=chroma_copy))
    answer, sources = rag.query("What is covered in lesson 1 of the MCP course?")
    assert answer.strip()
    assert (
        sources
    ), "a content question should search the course index and return sources"
