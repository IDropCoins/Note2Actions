from extraction import _map_span_to_chunk


def test_span_maps_to_correct_chunk():
    chunk_map = [
        {
            "chunk_id": "1",
            "source_file": "a.md",
            "meeting_date": "2026-02-14",
            "start": 0,
            "end": 20,
            "text": "This is chunk one!!!",
        }
    ]

    evidence = _map_span_to_chunk(chunk_map, 5, 9)

    assert evidence is not None
    assert evidence["chunk_id"] == "1"
    assert evidence["snippet"] == "is c"
    assert evidence["start_char"] == 5
    assert evidence["end_char"] == 9


def test_span_outside_chunk_returns_none():
    chunk_map = [
        {
            "chunk_id": "1",
            "source_file": "a.md",
            "meeting_date": "2026-02-14",
            "start": 0,
            "end": 10,
            "text": "Short text",
        }
    ]

    assert _map_span_to_chunk(chunk_map, 10, 15) is None


def test_negative_offsets_return_none():
    chunk_map = [
        {
            "chunk_id": "1",
            "source_file": "a.md",
            "meeting_date": "2026-02-14",
            "start": 0,
            "end": 10,
            "text": "Short text",
        }
    ]

    assert _map_span_to_chunk(chunk_map, -1, 5) is None


def test_context_len_guard():
    chunk_map = [
        {
            "chunk_id": "1",
            "source_file": "a.md",
            "meeting_date": "2026-02-14",
            "start": 0,
            "end": 100,
            "text": "x" * 100,
        }
    ]

    assert _map_span_to_chunk(chunk_map, 50, 200, context_len=100) is None
    assert _map_span_to_chunk(chunk_map, 50, 80, context_len=100) is not None
