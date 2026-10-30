"""Tests for CourseSearchTool.execute output."""

from conftest import MCP_COURSE

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def results(docs_and_meta):
    docs = [d for d, _ in docs_and_meta]
    meta = [m for _, m in docs_and_meta]
    return SearchResults(documents=docs, metadata=meta, distances=[0.1] * len(docs))


# ---------- Unit tests (mocked VectorStore) ----------


class TestExecuteWithMockStore:
    def test_forwards_arguments_to_store(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.execute(query="what is MCP", course_name="MCP", lesson_number=2)
        mock_store.search.assert_called_once_with(
            query="what is MCP", course_name="MCP", lesson_number=2
        )

    def test_formats_results_with_course_and_lesson_headers(self, mock_store):
        mock_store.search.return_value = results(
            [
                ("Servers expose tools.", {"course_title": "MCP", "lesson_number": 1}),
                ("Clients call tools.", {"course_title": "MCP", "lesson_number": 2}),
            ]
        )
        out = CourseSearchTool(mock_store).execute(query="tools")
        assert out == (
            "[MCP - Lesson 1]\nServers expose tools.\n\n"
            "[MCP - Lesson 2]\nClients call tools."
        )

    def test_header_omits_lesson_when_missing(self, mock_store):
        mock_store.search.return_value = results(
            [
                ("Course overview.", {"course_title": "MCP", "lesson_number": None}),
            ]
        )
        out = CourseSearchTool(mock_store).execute(query="overview")
        assert out == "[MCP]\nCourse overview."

    def test_returns_store_error_verbatim(self, mock_store):
        mock_store.search.return_value = SearchResults.empty("Search error: boom")
        assert CourseSearchTool(mock_store).execute(query="x") == "Search error: boom"

    def test_empty_results_message_without_filters(self, mock_store):
        assert (
            CourseSearchTool(mock_store).execute(query="x")
            == "No relevant content found."
        )

    def test_empty_results_message_includes_filters(self, mock_store):
        out = CourseSearchTool(mock_store).execute(
            query="x", course_name="MCP", lesson_number=3
        )
        assert out == "No relevant content found in course 'MCP' in lesson 3."

    def test_sources_are_deduplicated_and_linked(self, mock_store):
        mock_store.search.return_value = results(
            [
                ("a", {"course_title": "MCP", "lesson_number": 1}),
                ("b", {"course_title": "MCP", "lesson_number": 1}),
                ("c", {"course_title": "MCP", "lesson_number": None}),
            ]
        )
        tool = CourseSearchTool(mock_store)
        tool.execute(query="x")
        assert tool.last_sources == [
            {"label": "MCP - Lesson 1", "link": "https://example.com/MCP/lesson1"},
            {"label": "MCP", "link": "https://example.com/MCP"},
        ]

    def test_sources_accumulate_across_searches_without_duplicates(self, mock_store):
        # A query may run two searches; the UI should show sources from both.
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        mock_store.search.return_value = results(
            [("a", {"course_title": "A", "lesson_number": 1})]
        )
        manager.execute_tool("search_course_content", query="x")
        mock_store.search.return_value = results(
            [
                ("b", {"course_title": "B", "lesson_number": 2}),
                ("a2", {"course_title": "A", "lesson_number": 1}),
            ]
        )
        manager.execute_tool("search_course_content", query="y")
        assert [s["label"] for s in manager.get_last_sources()] == [
            "A - Lesson 1",
            "B - Lesson 2",
        ]

    def test_reset_sources_clears_accumulated_sources(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        mock_store.search.return_value = results(
            [("a", {"course_title": "A", "lesson_number": 1})]
        )
        manager.execute_tool("search_course_content", query="x")
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_tool_manager_dispatches_to_execute(self, mock_store):
        mock_store.search.return_value = results(
            [("a", {"course_title": "MCP", "lesson_number": 1})]
        )
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        out = manager.execute_tool(
            "search_course_content", query="x", course_name="MCP"
        )
        assert out.startswith("[MCP - Lesson 1]")
        assert manager.get_last_sources()[0]["label"] == "MCP - Lesson 1"


# ---------- Integration tests (copy of the real ChromaDB) ----------


class TestExecuteWithRealStore:
    def test_store_has_courses_indexed(self, real_store):
        assert real_store.get_course_count() > 0
        assert real_store.course_content.count() > 0

    def test_content_query_returns_formatted_results(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="What is the Model Context Protocol?")
        assert not out.startswith(("No relevant content", "Search error"))
        assert out.startswith("[")
        assert tool.last_sources

    def test_course_filter_restricts_results(self, real_store):
        out = CourseSearchTool(real_store).execute(
            query="servers and clients", course_name="MCP"
        )
        headers = [line for line in out.splitlines() if line.startswith("[")]
        assert headers
        assert all(h.startswith(f"[{MCP_COURSE}") for h in headers)

    def test_course_and_lesson_filter(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="introduction", course_name="MCP", lesson_number=1)
        headers = [line for line in out.splitlines() if line.startswith("[")]
        assert headers
        assert all(h == f"[{MCP_COURSE} - Lesson 1]" for h in headers)
        assert tool.last_sources[0][
            "link"
        ], "lesson link should be resolved from the catalog"
