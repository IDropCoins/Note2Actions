"""Structured extraction from meeting notes with grounded evidence."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dateparser
import langextract as lx
from langextract.data import CharInterval, ExampleData, Extraction

from config import Settings
from ingest.embedding import Embedder
from storage import ZvecStore

settings = Settings()
embedder = Embedder(settings.embed_model)
store = ZvecStore(settings.zvec_db_path)

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


def _norm_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized if normalized else None


def _norm_conf(attrs: Dict[str, Any]) -> str:
    # If the model provides confidence, pass it through unchanged.
    for key in ("confidence", "conf", "score"):
        value = attrs.get(key)
        if value is None:
            continue
        return str(value)
    return "unknown"


def _parse_due_date(raw: str, base_meeting_date: Optional[str]) -> Optional[str]:
    """
    Convert raw due date text into ISO date (YYYY-MM-DD) using meeting_date as base.
    base_meeting_date is expected as YYYY-MM-DD.
    """
    candidate = (raw or "").strip()
    if not candidate:
        return None

    base_dt = None
    if base_meeting_date:
        try:
            base_dt = datetime.fromisoformat(base_meeting_date)
        except ValueError:
            base_dt = None

    parsed = dateparser.parse(
        candidate,
        settings={
            "RELATIVE_BASE": base_dt or datetime.now(),
            "PREFER_DATES_FROM": "future",
        },
    )
    if not parsed:
        return None

    return parsed.date().isoformat()


def _as_decision(ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": ex_text,
        "confidence": _norm_conf(attrs),
        "evidence": evidence,
    }


def _as_action_item(
    ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]
) -> Dict[str, Any]:
    raw_due = _norm_str(attrs.get("due_date") or attrs.get("deadline"))
    normalized_due = _parse_due_date(raw_due, _norm_str(evidence.get("meeting_date")))
    return {
        "task": _norm_str(attrs.get("task")) or ex_text,
        "owner": _norm_str(attrs.get("owner")),
        "due_date": normalized_due,
        "due_date_raw": raw_due,
        "priority": _norm_str(attrs.get("priority")),
        "confidence": _norm_conf(attrs),
        "evidence": evidence,
    }


def _as_deadline(ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    raw_due = _norm_str(attrs.get("due_date") or attrs.get("deadline"))
    normalized_due = _parse_due_date(raw_due, _norm_str(evidence.get("meeting_date")))
    return {
        "label": _norm_str(attrs.get("label")) or ex_text,
        "due_date": normalized_due,
        "due_date_raw": raw_due,
        "confidence": _norm_conf(attrs),
        "evidence": evidence,
    }


def _as_open_question(
    ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "question": _norm_str(attrs.get("question")) or ex_text,
        "owner": _norm_str(attrs.get("owner")),
        "confidence": _norm_conf(attrs),
        "evidence": evidence,
    }


def _as_risk(ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "risk": _norm_str(attrs.get("risk")) or ex_text,
        "severity": _norm_str(attrs.get("severity")),
        "confidence": _norm_conf(attrs),
        "evidence": evidence,
    }


_CLASS_TO_BUCKET = {
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


def _build_bucket_item(
    bucket: str, ex_text: str, attrs: Dict[str, Any], evidence: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if bucket == "decisions":
        return _as_decision(ex_text, attrs, evidence)
    if bucket == "action_items":
        return _as_action_item(ex_text, attrs, evidence)
    if bucket == "deadlines":
        return _as_deadline(ex_text, attrs, evidence)
    if bucket == "open_questions":
        return _as_open_question(ex_text, attrs, evidence)
    if bucket == "risks":
        return _as_risk(ex_text, attrs, evidence)
    return None


def _normalize_extraction(
    extraction: Any, chunk_map: List[dict]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    extraction_type = (_get_attr(extraction, "extraction_class", "") or "").strip().lower()
    bucket = _CLASS_TO_BUCKET.get(extraction_type)
    if bucket is None:
        return None

    ex_text = _get_attr(extraction, "extraction_text", "") or ""
    ex_attrs_raw = _get_attr(extraction, "attributes", {}) or {}
    ex_attrs = ex_attrs_raw if isinstance(ex_attrs_raw, dict) else {}
    global_start, global_end = _resolve_offsets(extraction)

    if not isinstance(global_start, int) or not isinstance(global_end, int):
        return None
    evidence = _map_span_to_chunk(chunk_map, global_start, global_end)
    if evidence is None:
        return None

    item = _build_bucket_item(bucket, ex_text, ex_attrs, evidence)
    if item is None:
        return None
    return bucket, item


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
        normalized = _normalize_extraction(ex, chunk_map)
        if normalized is None:
            continue

        bucket, item = normalized
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
