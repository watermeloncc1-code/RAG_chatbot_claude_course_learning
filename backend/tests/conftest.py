import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

from config import config
from vector_store import VectorStore, SearchResults

BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BACKEND_DIR / "chroma_db"

MCP_COURSE = "MCP: Build Rich-Context AI Apps with Anthropic"


# ---------- Fake Anthropic responses ----------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block(thinking=""):
    return SimpleNamespace(type="thinking", thinking=thinking, signature="sig")


def tool_use_block(name, tool_input, block_id="toolu_01"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def make_response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def text_response(text):
    return make_response([text_block(text)])


def tool_use_response(name, tool_input, block_id="toolu_01"):
    return make_response([tool_use_block(name, tool_input, block_id)], stop_reason="tool_use")


# ---------- Vector store fixtures ----------

@pytest.fixture
def mock_store():
    """VectorStore double; set mock_store.search.return_value per test."""
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.side_effect = lambda title, n: f"https://example.com/{title}/lesson{n}"
    store.get_course_link.side_effect = lambda title: f"https://example.com/{title}"
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    return store


@pytest.fixture(scope="session")
def chroma_copy(tmp_path_factory):
    """Copy of the real ChromaDB so tests never touch the running server's index."""
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        pytest.skip("backend/chroma_db not built; start the server once to ingest docs/")
    dest = tmp_path_factory.mktemp("chroma") / "chroma_db"
    shutil.copytree(CHROMA_DIR, dest)
    return str(dest)


@pytest.fixture(scope="session")
def real_store(chroma_copy):
    return VectorStore(chroma_copy, config.EMBEDDING_MODEL, config.MAX_RESULTS)


@pytest.fixture
def require_api_key():
    if not config.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY is not set in .env")


# ---------- API test app ----------
# backend/app.py builds a real RAGSystem and mounts ../frontend at import time,
# so the API tests use this copy instead. Keep models and endpoints in sync with app.py.

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    label: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system, static_dir):
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


INDEX_HTML = "<!DOCTYPE html><html><body><h1>Course Materials Assistant</h1></body></html>"


@pytest.fixture
def mock_rag_system():
    """RAGSystem double; override query/get_course_analytics per test."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.query.return_value = (
        "Lesson 1 introduces MCP.",
        [{"label": f"{MCP_COURSE} - Lesson 1", "link": "https://example.com/lesson1"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": [MCP_COURSE, "Other Course"],
    }
    return rag


@pytest.fixture
def static_dir(tmp_path):
    """Stand-in for frontend/."""
    (tmp_path / "index.html").write_text(INDEX_HTML)
    return str(tmp_path)


@pytest.fixture
def api_client(mock_rag_system, static_dir):
    with TestClient(create_test_app(mock_rag_system, static_dir)) as client:
        yield client
