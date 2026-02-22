"""Structured extraction from meeting notes with grounded evidence."""

from typing import Any, Dict, List, Optional, Tuple

import langextract as lx
from langextract.data import CharInterval, ExampleData, Extraction

from config import Settings
from ingest.embedding import Embedder
from storage import ZvecStore

settings = Settings()
embedder = Embedder(settings.embed_model)
store = ZvecStore(settings.zvec_db_path)

_TYPE_TO_BUCKET = {
    "decision": "decisions",
    "decisions": "decisions",
    "action_item": "action_items",
    "action items": "action_items",
    "actionitem": "action_items",
    "task": "action_items",
    "deadline": "deadlines",
    "deadlines": "deadlines",
    "due_date": "deadlines",
    "due date": "deadlines",
    "open_question": "open_questions",
    "open questions": "open_questions",
    "question": "open_questions",
    "risk": "risks",
    "risks": "risks",
}

_EXAMPLE_TEXT = (
    "## Decisions\n"
    "- Keep the current signup copy for one more week.\n\n"
    "## Action Items\n"
    "- Priya: share final onboarding metrics by Monday.\n\n"
    "## Risks\n"
    "- Third-party API rate limits may affect daily sync jobs."
)

EXTRACTION_EXAMPLES = [
    ExampleData(
        text=_EXAMPLE_TEXT,
        extractions=[
            Extraction(
                extraction_class="decision",
                extraction_text="Keep the current signup copy for one more week.",
                char_interval=CharInterval(start_pos=18, end_pos=65),
            ),
            Extraction(
                extraction_class="action_item",
                extraction_text="Priya: share final onboarding metrics by Monday.",
                char_interval=CharInterval(start_pos=86, end_pos=134),
                attributes={"owner": "Priya", "deadline": "Monday"},
            ),
            Extraction(
                extraction_class="risk",
                extraction_text="Third-party API rate limits may affect daily sync jobs.",
                char_interval=CharInterval(start_pos=146, end_pos=201),
            ),
        ],
    )
]


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    return dict(doc.fields) if getattr(doc, "fields", None) else {}


def _build_context_with_map(search_results: List[Any]) -> Tuple[str, List[dict]]:
    """
    Build a combined context string and chunk offset map.

    The map lets us translate LangExtract global offsets back to chunk-local offsets.
    """
    parts: List[str] = []
    chunk_map: List[dict] = []
    cursor = 0

    for doc in search_results:
        metadata = _doc_metadata(doc)
        text = metadata.get("text", "") or ""

        # Keep separator stable so offsets remain deterministic.
        sep = "\n\n" if parts else ""
        cursor += len(sep)
        parts.append(sep + text)

        start = cursor
        end = start + len(text)

        chunk_map.append(
            {
                "chunk_id": getattr(doc, "id", None) or metadata.get("chunk_id"),
                "source_file": metadata.get("source_file"),
                "meeting_date": metadata.get("meeting_date"),
                "start": start,
                "end": end,
                "text": text,
            }
        )

        cursor = end

    combined_context = "".join(parts)
    return combined_context, chunk_map


def _map_span_to_chunk(chunk_map: List[dict], start: int, end: int) -> Optional[dict]:
    """Return evidence if a span fits fully within one chunk; otherwise return None."""
    if start is None or end is None:
        return None
    if start < 0 or end < 0 or end <= start:
        return None

    for chunk in chunk_map:
        if start >= chunk["start"] and end <= chunk["end"]:
            local_start = start - chunk["start"]
            local_end = end - chunk["start"]
            snippet = chunk["text"][local_start:local_end]

            return {
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "meeting_date": chunk["meeting_date"],
                "start_char": local_start,
                "end_char": local_end,
                "snippet": snippet,
            }

    return None


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Handle both object attributes and dict-like shapes safely."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _resolve_offsets(extraction: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Support both LangExtract offset shapes:
    - start_char / end_char
    - char_interval.start_pos / char_interval.end_pos
    """
    start = _get_attr(extraction, "start_char", None)
    end = _get_attr(extraction, "end_char", None)
    if isinstance(start, int) and isinstance(end, int):
        return start, end

    interval = _get_attr(extraction, "char_interval", None)
    if interval is None:
        return None, None

    interval_start = _get_attr(interval, "start_pos", None)
    interval_end = _get_attr(interval, "end_pos", None)
    if isinstance(interval_start, int) and isinstance(interval_end, int):
        return interval_start, interval_end

    return None, None


def _iter_extractions(result: Any) -> List[Any]:
    if isinstance(result, list):
        items: List[Any] = []
        for annotated_doc in result:
            items.extend(getattr(annotated_doc, "extractions", []) or [])
        return items
    return getattr(result, "extractions", []) or []


def _dedupe_search_results(search_results: List[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[Tuple[str, str, str]] = set()

    for doc in search_results:
        metadata = _doc_metadata(doc)
        key = (
            str(metadata.get("source_file") or ""),
            str(metadata.get("meeting_date") or ""),
            str(metadata.get("text") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    return deduped


def _dedupe_bucket_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int, int]] = set()

    for item in items:
        evidence = item.get("evidence") or {}
        key = (
            str(evidence.get("source_file") or ""),
            str(evidence.get("snippet") or "").strip(),
            int(evidence.get("start_char") or 0),
            int(evidence.get("end_char") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def extract_actions(query: str, top_k: int = 5) -> Dict[str, Any]:
    # 1) Search
    query_vector = embedder.embed([query])[0]
    search_results = _dedupe_search_results(store.search(query_vector, top_k=top_k))

    # 2) Build combined context + offset map
    combined_context, chunk_map = _build_context_with_map(search_results)

    # 3) LangExtract call
    prompt = """
    Extract the following from the text, only when clearly supported:

    - decisions
    - action items
    - owners
    - deadlines
    - open questions
    - risks

    For each extraction, include exact grounding start_char and end_char offsets.
    Do not hallucinate missing fields.
    """

    result = lx.extract(
        text_or_documents=combined_context,
        prompt_description=prompt,
        examples=EXTRACTION_EXAMPLES,
        model_id="gemini-2.5-flash",
    )

    # 4) Normalize into API shape with chunk-level evidence
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "decisions": [],
        "action_items": [],
        "deadlines": [],
        "open_questions": [],
        "risks": [],
    }

    for ex in _iter_extractions(result):
        ex_type = _get_attr(ex, "extraction_class", "") or ""
        ex_text = _get_attr(ex, "extraction_text", "") or ""
        ex_attrs = _get_attr(ex, "attributes", {}) or {}
        global_start, global_end = _resolve_offsets(ex)

        evidence = None
        if isinstance(global_start, int) and isinstance(global_end, int):
            evidence = _map_span_to_chunk(chunk_map, global_start, global_end)

        # Keep output strictly grounded.
        if evidence is None:
            continue

        item: Dict[str, Any] = {
            "text": ex_text,
            "attributes": ex_attrs,
            "evidence": evidence,
        }

        extraction_type = ex_type.strip().lower()
        bucket = _TYPE_TO_BUCKET.get(extraction_type)
        if bucket:
            grouped[bucket].append(item)

    for bucket_name, bucket_items in grouped.items():
        grouped[bucket_name] = _dedupe_bucket_items(bucket_items)

    return {
        "query": query,
        "top_k": top_k,
        "results": grouped,
        "retrieved_chunks": [
            {
                "chunk_id": getattr(doc, "id", None) or _doc_metadata(doc).get("chunk_id"),
                "source_file": _doc_metadata(doc).get("source_file"),
                "meeting_date": _doc_metadata(doc).get("meeting_date"),
                "score": getattr(doc, "score", None),
            }
            for doc in search_results
        ],
    }
