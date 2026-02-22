from datetime import date

from ingest.core import chunk_document


def test_chunk_indices_are_ordered():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    _doc_id, chunks = chunk_document(
        source_file="test.md",
        text=text,
        meeting_date=date(2026, 2, 14),
        max_chars=1000,
        overlap_chars=0,
    )

    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)


def test_all_chunks_have_text():
    text = "# Heading\n\n- Item one\n- Item two\n\n## Section\n\nBody text here."
    _doc_id, chunks = chunk_document(
        source_file="test.md",
        text=text,
        meeting_date=date(2026, 2, 14),
        max_chars=1000,
        overlap_chars=0,
    )

    for chunk in chunks:
        assert chunk.text.strip(), f"chunk_index={chunk.chunk_index} has empty text"
