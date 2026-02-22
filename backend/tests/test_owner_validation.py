from extraction import _owner_is_grounded


def test_owner_in_snippet_is_grounded():
    evidence = {
        "chunk_id": "1",
        "snippet": "Priya: share final onboarding metrics by Monday.",
    }

    assert _owner_is_grounded("Priya", evidence) is True


def test_owner_not_in_text_is_not_grounded():
    evidence = {
        "chunk_id": "1",
        "snippet": "Send proposal by Friday",
    }

    assert _owner_is_grounded("John", evidence) is False


def test_none_owner_is_not_grounded():
    evidence = {
        "chunk_id": "1",
        "snippet": "Some text here",
    }

    assert _owner_is_grounded(None, evidence) is False


def test_owner_case_insensitive():
    evidence = {
        "chunk_id": "1",
        "snippet": "KRISH will handle the release.",
    }

    assert _owner_is_grounded("krish", evidence) is True
