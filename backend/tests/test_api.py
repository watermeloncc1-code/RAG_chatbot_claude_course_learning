"""Tests for the FastAPI endpoints, run against the test app in conftest.py."""
import pytest

from conftest import INDEX_HTML, MCP_COURSE

pytestmark = pytest.mark.api

QUESTION = "What is covered in lesson 1 of the MCP course?"


class TestQueryEndpoint:
    def test_returns_answer_sources_and_session(self, api_client):
        resp = api_client.post("/api/query", json={"query": QUESTION})

        assert resp.status_code == 200
        assert resp.json() == {
            "answer": "Lesson 1 introduces MCP.",
            "sources": [{"label": f"{MCP_COURSE} - Lesson 1", "link": "https://example.com/lesson1"}],
            "session_id": "session_1",
        }

    def test_creates_session_when_missing(self, api_client, mock_rag_system):
        api_client.post("/api/query", json={"query": QUESTION})

        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with(QUESTION, "session_1")

    def test_reuses_given_session(self, api_client, mock_rag_system):
        resp = api_client.post("/api/query", json={"query": QUESTION, "session_id": "session_7"})

        assert resp.json()["session_id"] == "session_7"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(QUESTION, "session_7")

    def test_source_without_link_serializes_as_null(self, api_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer.", [{"label": MCP_COURSE}])

        resp = api_client.post("/api/query", json={"query": QUESTION})

        assert resp.json()["sources"] == [{"label": MCP_COURSE, "link": None}]

    def test_missing_query_is_rejected(self, api_client, mock_rag_system):
        resp = api_client.post("/api/query", json={"session_id": "session_1"})

        assert resp.status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_non_json_body_is_rejected(self, api_client, mock_rag_system):
        resp = api_client.post(
            "/api/query", content="not json", headers={"Content-Type": "application/json"}
        )

        assert resp.status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_rag_error_becomes_500_with_detail(self, api_client, mock_rag_system):
        mock_rag_system.query.side_effect = Exception("boom")

        resp = api_client.post("/api/query", json={"query": QUESTION})

        assert resp.status_code == 500
        assert resp.json() == {"detail": "boom"}


class TestCoursesEndpoint:
    def test_returns_course_stats(self, api_client):
        resp = api_client.get("/api/courses")

        assert resp.status_code == 200
        assert resp.json() == {"total_courses": 2, "course_titles": [MCP_COURSE, "Other Course"]}

    def test_empty_catalog(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}

        resp = api_client.get("/api/courses")

        assert resp.json() == {"total_courses": 0, "course_titles": []}

    def test_analytics_error_becomes_500_with_detail(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("chroma unavailable")

        resp = api_client.get("/api/courses")

        assert resp.status_code == 500
        assert resp.json() == {"detail": "chroma unavailable"}


class TestSessionEndpoint:
    def test_clears_session(self, api_client, mock_rag_system):
        resp = api_client.delete("/api/session/session_1")

        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("session_1")


class TestStaticRoot:
    def test_root_serves_index_html(self, api_client):
        resp = api_client.get("/")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.text == INDEX_HTML

    def test_unknown_path_is_404(self, api_client):
        assert api_client.get("/missing.js").status_code == 404

    def test_api_routes_take_precedence_over_mount(self, api_client):
        resp = api_client.get("/api/courses")

        assert resp.headers["content-type"] == "application/json"
