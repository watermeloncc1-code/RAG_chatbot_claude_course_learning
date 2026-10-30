# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (Python 3.13+). Requires `ANTHROPIC_API_KEY` in `.env` at the repo root.

**Always use `uv` for dependencies and running Python. Never use `pip`, `pip install`, `python -m venv`, or a bare `python`/`python3`.** `pyproject.toml` and `uv.lock` are the source of truth; there is no `requirements.txt`.

```bash
uv sync                     # install dependencies from uv.lock
uv add <package>            # add a dependency (updates pyproject.toml and uv.lock)
uv remove <package>         # remove a dependency
uv run <command>            # run anything in the project env, e.g. uv run python script.py
./run.sh                    # start the server (cd backend && uv run uvicorn app:app --reload --port 8000)
```

- Web UI: http://localhost:8000 — API docs: http://localhost:8000/docs
- The server must be started from `backend/`: paths are relative to it (`../docs`, `../frontend`, `./chroma_db`).
- `.claude/launch.json` defines a `ragchatbot` config (same uvicorn command via `uv run --directory backend`) for the desktop app's browser preview. `uv` is installed at `~/.local/bin/uv`, which may not be on PATH in non-login shells.
- The first startup downloads the `all-MiniLM-L6-v2` embedding model and embeds all of `docs/` (528 chunks across 4 courses), so it can take a minute before requests are served. Later startups reuse `backend/chroma_db/`.
- `--reload` watches only `backend/`. Editing `.env` (repo root) or `docs/` does not trigger a reload; restart the server.
- Tests: `uv run pytest` from the repo root (pytest config in `pyproject.toml`, tests in `backend/tests/`). Tests marked `live` call the real API with the `.env` key; skip them with `uv run pytest -m 'not live'`. Integration tests run against a temp copy of `backend/chroma_db/`.
- There is no linter or build step. The root `main.py` is an unused placeholder.

## Architecture

A RAG chatbot over course transcripts. FastAPI (`backend/app.py`) serves both the JSON API and the static frontend (`frontend/`, vanilla JS mounted at `/`). Two endpoints: `POST /api/query` and `GET /api/courses`.

| File | Role |
|---|---|
| `frontend/script.js` | `sendMessage()` POSTs `{query, session_id}` to `/api/query`, renders the Markdown answer with `marked`, and shows sources in a collapsible block. `loadCourseStats()` calls `/api/courses` for the sidebar. |
| `backend/app.py` | FastAPI app, request/response models (`QueryRequest`, `QueryResponse`, `CourseStats`), startup ingestion, static mount. Any exception becomes HTTP 500 with the message in `detail`. |
| `backend/rag_system.py` | `RAGSystem`: wires all components together; `query()` is the per-request entry point, `add_course_folder()` does ingestion. |
| `backend/ai_generator.py` | `AIGenerator`: Anthropic client, `SYSTEM_PROMPT`, the bounded tool loop (up to `MAX_TOOL_ROUNDS` = 2 rounds, at most 3 API calls). |
| `backend/search_tools.py` | `Tool` ABC, `CourseSearchTool` (`search_course_content`), `ToolManager` registry/dispatcher. |
| `backend/vector_store.py` | `VectorStore` (ChromaDB wrapper) and `SearchResults` dataclass. |
| `backend/document_processor.py` | Parses course files and chunks lesson text. |
| `backend/session_manager.py` | In-memory conversation history keyed by `session_N` IDs. |
| `backend/models.py` | Pydantic `Course`, `Lesson`, `CourseChunk`. |
| `backend/config.py` | `Config` dataclass; loads `.env` via `python-dotenv`. |

Request path for a course question: `script.js` → `app.py:query_documents` → `RAGSystem.query` (adds history) → `AIGenerator.generate_response` (Claude call 1, returns `tool_use`) → `ToolManager.execute_tool` → `CourseSearchTool.execute` → `VectorStore.search` (resolve course name in `course_catalog`, then filtered query of `course_content`) → Claude call 2 with the tool results (Claude may search once more, then call 3 answers) → answer + `last_sources` returned to the browser. General-knowledge questions stop after call 1.

**Tool-based retrieval, not retrieve-then-generate.** `RAGSystem.query()` does not search up front. It hands Claude a single tool (`search_course_content`, defined in `search_tools.py`) and Claude decides whether to call it. `AIGenerator.generate_response()` runs a loop: while the response has `stop_reason == "tool_use"` and fewer than `MAX_TOOL_ROUNDS` (2) rounds have run, `_execute_tools()` runs every tool_use block via `ToolManager`, appends the assistant content (verbatim, including thinking blocks) and one user message of tool_results, and calls Claude again. The call after the 2nd round, or after any tool raises, uses `tool_choice: none` so Claude must answer. A tool exception becomes a `tool_result` with `is_error: True` (`Tool error: ...`) instead of an HTTP 500. Several tool_use blocks in one response count as one round.

**Claude call settings** (`AIGenerator`): model from `config.ANTHROPIC_MODEL`, `max_tokens: 4096`, no `temperature` (current models reject non-default values). Every call passes `tools` (the API rejects tool_use/tool_result blocks without them); `tool_choice` is `auto` while more rounds are allowed, otherwise `none`. Behavior rules (search only for course-specific questions, up to 2 searches, no meta-commentary, brief answers) live in `AIGenerator.SYSTEM_PROMPT`; change answer style there. The final response goes through `_extract_text()`, which joins the text blocks and skips thinking/tool_use blocks.

**Search tool arguments**: `query` (required), `course_name` (partial names like "MCP" work), `lesson_number` (int). `VectorStore._build_filter()` turns these into a Chroma `where` clause (`$and` when both filters are set).

**Sources flow through tool state.** `CourseSearchTool._format_results()` appends source labels to `self.last_sources` (deduplicated, so both searches of a 2-round query contribute); after generation, `RAGSystem` reads them via `ToolManager.get_last_sources()` and then calls `reset_sources()`. New tools that should surface sources need a `last_sources` attribute.

**Two ChromaDB collections** (`vector_store.py`, embeddings via `all-MiniLM-L6-v2`):
- `course_catalog` — one doc per course, ID = course title; metadata includes `lessons_json` (serialized lesson list). Used to fuzzy-resolve a `course_name` argument to an exact title via vector search.
- `course_content` — text chunks with `course_title`, `lesson_number`, `chunk_index` metadata; IDs are `{title_with_underscores}_{chunk_index}`. `search()` resolves the course name, builds a `where` filter, then queries this collection.

**Ingestion** runs on FastAPI startup: `add_course_folder("../docs")` processes `.txt/.pdf/.docx` files and skips courses whose title is already in the catalog (course title is the unique key, so re-ingesting a changed file requires clearing `backend/chroma_db/`). `DocumentProcessor` expects this file format:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<transcript text...>
```

Lesson text is split into sentence-based chunks (`CHUNK_SIZE` 800 chars, `CHUNK_OVERLAP` 100; overlap is whole sentences fitting in 100 chars) and prefixed with lesson context before embedding.

**Sessions** (`session_manager.py`) are in-memory only and lost on restart. History is kept to `MAX_HISTORY` exchanges and injected into the system prompt as text, not as message turns. The frontend gets a `session_id` from the first `/api/query` response and sends it on later requests.

All tunables (model name, chunk sizes, result count, history length, Chroma path) live in `backend/config.py`.

## Debugging

- The frontend shows only `Error: Query failed` for any non-2xx response. The real error is in the response body's `detail`; reproduce with:
  ```bash
  curl -s -X POST http://localhost:8000/api/query -H 'Content-Type: application/json' -d '{"query":"hi"}'
  ```
- `401 invalid x-api-key` means `.env` still holds the placeholder from `.env.example` or an invalid key. Fix the key, then restart the server (see the reload note above).
- To rebuild the vector index (changed chunking, edited a course file), stop the server and delete `backend/chroma_db/`; ingestion reruns on next startup.
- `huggingface/tokenizers ... forked` warnings in the server log are harmless.

## Known quirks

- Chunk prefixes are inconsistent in `process_course_document()`: in all lessons except the last, only the first chunk gets `Lesson N content: `; in the final lesson, every chunk gets `Course <title> Lesson N content: `.
- `.pdf` and `.docx` files are accepted by `add_course_folder()` but read as UTF-8 text, so they ingest as garbage. Only `.txt` works.
- `_resolve_course_name()` always returns the nearest course (no distance threshold), so a `course_name` that matches nothing still filters to some course.
- `DevStaticFiles` (no-cache headers) is defined in `app.py` but unused; the mount uses plain `StaticFiles`. `script.js` and `style.css` are cache-busted with `?v=N` in `index.html` instead.
