import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
