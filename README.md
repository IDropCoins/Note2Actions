# Note2Action

Note2Action is a privacy-first, local-only meeting intelligence backend.

It ingests raw meeting notes and converts them into:
- Decisions
- Action items
- Deadlines
- Open questions
- Risks

All extracted outputs are grounded to exact evidence spans.

## What It Does

### 1) Ingestion
- Reads `.md` and `.txt` notes
- Infers meeting dates (filename, content, then mtime fallback)
- Splits content into deterministic chunks
- Generates embeddings with `sentence-transformers`
- Stores vectors locally in Zvec
- Uses manifest-based idempotency to skip unchanged files and prevent duplicates

### 2) Semantic Search
- Meaning-based retrieval (not keyword matching)
- Supports metadata filters:
  - `source_file`
  - `date_from`
  - `date_to`

### 3) Structured Extraction
- Uses LangExtract (Gemini-backed)
- Extracts only from retrieved chunks
- Returns strict schema buckets:
  - `decisions`
  - `action_items`
  - `deadlines`
  - `open_questions`
  - `risks`
- Normalizes relative due dates to ISO (`YYYY-MM-DD`)
- Rejects hallucinated owners (`owner` must appear in evidence text)
- Grounds each item to evidence:
  - `chunk_id`
  - `start_char`
  - `end_char`
  - `snippet`

## Architecture

```text
Raw Notes
  -> Chunking
  -> Embeddings
  -> Zvec (local vector store)
  -> Semantic Retrieval
  -> LangExtract
  -> Grounded Structured Output
```

Backend layout:

```text
backend/
  ingest/
    core.py
    embedding.py
    manifest.py
  storage.py
  extraction.py
  api.py
  config.py
```

## Prerequisites

| Dependency | Required For |
|---|---|
| Python 3.11+ (`.python-version` pins `3.11.8`) | Runtime |
| [Zvec](https://github.com/alibaba/zvec) | Local vector storage + similarity search |
| [LangExtract](https://pypi.org/project/langextract/) | Structured extraction with grounding |
| Gemini API key (`LANGEXTRACT_API_KEY`) | LangExtract model calls |

### Zvec (Quick Context)

Zvec is an open-source, embedded vector database from Alibaba Tongyi Lab. It is designed to run in-process (SQLite-style), so you do not need a separate vector DB server. In this project, Zvec stores chunk embeddings and metadata locally on disk (`./zvec_db`), and powers retrieval for both `/search` and `/extract`.

## Installation

```bash
# From project root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# set LANGEXTRACT_API_KEY in .env
```

## Ingest Notes

Place notes in `sample_notes/`, then run:

```bash
# from project root
python -m backend.ingest --path ./sample_notes
```

Useful options:

```bash
python -m backend.ingest --path ./sample_notes --rebuild
python -m backend.ingest --path ./sample_notes --batch-size 32
```

Manifest path:

```text
zvec_db/manifest.json
```

Unchanged files are skipped automatically when `--rebuild` is not set.

## Run API

```bash
# from project root
uvicorn main:app --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

## API Examples

### `POST /search`

```json
{
  "query": "budget approval",
  "top_k": 5,
  "filters": {
    "source_file": "meeting1.md",
    "date_from": "2026-02-01",
    "date_to": "2026-02-28"
  }
}
```

### `POST /extract`

```json
{
  "query": "proposal deadline",
  "top_k": 8
}
```

Response includes structured categories, normalized due dates, owner validation, and grounded evidence.

## Testing

Run all backend tests:

```bash
.venv/bin/python -m pytest backend/tests/ -v
```

Coverage includes:
- Date inference fallback
- Chunking determinism
- Grounding span mapping
- Due date normalization
- Owner validation

## Privacy

- No external vector DB
- No cloud vector storage
- Local Zvec only

## Project Status

- Backend Core: Complete
- Semantic Search: Complete
- Grounded Extraction: Complete
- Production Hardening: Implemented
- Frontend: Optional

## Why Note2Action

Many note tools summarize loosely and lose traceability.

Note2Action follows a stricter flow:
1. Retrieve relevant evidence first
2. Extract structured items second
3. Keep everything grounded
4. Avoid fabricated owners or dates

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software with attribution. See the [LICENSE](LICENSE) file for full details.

## Author 👤

**Shivay Bajaj**

- GitHub: https://github.com/IDropCoins
