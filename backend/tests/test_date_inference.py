from datetime import datetime

from ingest.core import infer_meeting_date


def test_date_fallback_to_mtime():
    text = "Lorem ipsum dolor sit amet."
    mtime = datetime(2026, 2, 20).timestamp()

    result = infer_meeting_date(
        relative_path="note.md",
        text=text,
        file_mtime=mtime,
    )

    assert result.isoformat() == "2026-02-20"


def test_date_from_filename():
    result = infer_meeting_date(
        relative_path="2026-03-15_standup.md",
        text="No date in body.",
        file_mtime=datetime(2026, 1, 1).timestamp(),
    )

    assert result.isoformat() == "2026-03-15"
