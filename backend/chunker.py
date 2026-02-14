import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Tuple


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
