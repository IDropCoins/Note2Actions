from extraction import _parse_due_date


def test_relative_due_date_with_base():
    result = _parse_due_date(raw="Friday", base_meeting_date="2026-02-18")

    assert result is not None
    assert len(result) == 10  # YYYY-MM-DD


def test_empty_raw_returns_none():
    assert _parse_due_date(raw="", base_meeting_date="2026-02-18") is None
    assert _parse_due_date(raw=None, base_meeting_date="2026-02-18") is None


def test_unparseable_returns_none():
    assert _parse_due_date(raw="xyzzy gibberish", base_meeting_date="2026-02-18") is None


def test_absolute_date():
    result = _parse_due_date(raw="March 5, 2026", base_meeting_date="2026-02-18")

    assert result == "2026-03-05"
