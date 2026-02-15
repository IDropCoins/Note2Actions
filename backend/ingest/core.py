import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import dateparser
from dateparser.search import search_dates

ALLOWED_EXTENSIONS = {".md", ".txt"}
FILENAME_DATE_PATTERN = re.compile(
    r"(\d{4}[-_/]\d{2}[-_/]\d{2})|(\d{2}[-_/]\d{2}[-_/]\d{4})"
)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    source_file: str
    meeting_date: date
    chunk_index: int
    text: str
    created_at: datetime
    tags: List[str]


def discover_note_files(root_path: str) -> List[Tuple[str, str]]:
    """
    Recursively discover .md and .txt files.

    Returns:
        List of tuples:
            (absolute_path, relative_path_from_root)
    """
    root = Path(root_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    results = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            abs_path = str(path.resolve())
            rel_path = str(path.relative_to(root))
            results.append((abs_path, rel_path))

    # Stable deterministic ordering
    results.sort(key=lambda x: x[1].lower())
    return results


def load_text(file_path: str) -> str:
    """
    Safely load a text file and normalize formatting.

    - Reads as UTF-8 with fallback replacement
    - Normalizes Windows/Mac newlines to 

    - Strips trailing whitespace on each line
    - Preserves paragraph structure
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _parse_date(text: str) -> Optional[date]:
    parsed = dateparser.parse(text)
    if parsed:
        return parsed.date()
    return None


def infer_meeting_date(relative_path: str, text: str, file_mtime: float) -> date:
    """
    Infer meeting date using:
    1. Filename patterns
    2. First 20 lines of content
    3. File modified timestamp fallback
    """
    filename_match = FILENAME_DATE_PATTERN.search(relative_path)
    if filename_match:
        parsed = _parse_date(filename_match.group(0))
        if parsed:
            return parsed

    first_lines = "\n".join(text.split("\n")[:20])
    found_dates = search_dates(first_lines)
    if found_dates:
        for _, parsed_datetime in found_dates:
            return parsed_datetime.date()

    return datetime.fromtimestamp(file_mtime).date()


def _split_markdown_sections(text: str) -> List[str]:
    sections = []
    current = []

    for line in text.split("\n"):
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return sections


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _make_chunk(
    *,
    doc_id: str,
    source_file: str,
    meeting_date: date,
    chunk_index: int,
    text: str,
    created_at: datetime,
) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=str(uuid.uuid4()),
        source_file=source_file,
        meeting_date=meeting_date,
        chunk_index=chunk_index,
        text=text.strip(),
        created_at=created_at,
        tags=[],
    )


def _build_paragraph_chunks(
    *,
    para: str,
    doc_id: str,
    source_file: str,
    meeting_date: date,
    chunk_index: int,
    created_at: datetime,
    max_chars: int,
    overlap_chars: int,
) -> Tuple[List[Chunk], int]:
    chunks: List[Chunk] = []
    start = 0
    step = max(1, max_chars - overlap_chars)

    while start < len(para):
        end = start + max_chars
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                source_file=source_file,
                meeting_date=meeting_date,
                chunk_index=chunk_index,
                text=para[start:end],
                created_at=created_at,
            )
        )
        chunk_index += 1
        start += step

    return chunks, chunk_index


def chunk_document(
    source_file: str,
    text: str,
    meeting_date: date,
    max_chars: int,
    overlap_chars: int,
) -> Tuple[str, List[Chunk]]:
    doc_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    if source_file.lower().endswith(".md"):
        sections = _split_markdown_sections(text)
    else:
        sections = [text]

    paragraphs = []
    for section in sections:
        paragraphs.extend(_split_paragraphs(section))

    chunks: List[Chunk] = []
    buffer = ""
    chunk_index = 0

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer += ("\n\n" if buffer else "") + para
            continue

        if buffer:
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    source_file=source_file,
                    meeting_date=meeting_date,
                    chunk_index=chunk_index,
                    text=buffer,
                    created_at=created_at,
                )
            )
            chunk_index += 1

        if len(para) > max_chars:
            para_chunks, chunk_index = _build_paragraph_chunks(
                para=para,
                doc_id=doc_id,
                source_file=source_file,
                meeting_date=meeting_date,
                chunk_index=chunk_index,
                created_at=created_at,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            chunks.extend(para_chunks)
            buffer = ""
        else:
            buffer = para

    if buffer:
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                source_file=source_file,
                meeting_date=meeting_date,
                chunk_index=chunk_index,
                text=buffer,
                created_at=created_at,
            )
        )

    return doc_id, chunks
