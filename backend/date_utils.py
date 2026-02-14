import re
from datetime import date, datetime
from typing import Optional

import dateparser
from dateparser.search import search_dates


FILENAME_DATE_PATTERN = re.compile(
    r"(\d{4}[-_/]\d{2}[-_/]\d{2})|(\d{2}[-_/]\d{2}[-_/]\d{4})"
)


def _parse_date(text: str) -> Optional[date]:
    parsed = dateparser.parse(text)
    if parsed:
        return parsed.date()
    return None


def infer_meeting_date(
    relative_path: str,
    text: str,
    file_mtime: float,
) -> date:
    """
    Infer meeting date using:
    1. Filename patterns
    2. First 20 lines of content
    3. File modified timestamp fallback
    """

    # 1) Try filename
    filename_match = FILENAME_DATE_PATTERN.search(relative_path)
    if filename_match:
        parsed = _parse_date(filename_match.group(0))
        if parsed:
            return parsed

    # 2) Try first 20 lines
    first_lines = "\n".join(text.split("\n")[:20])

    # Look for lines like:
    # Date: Feb 14 2026
    # ## 14 Feb 2026
    # # Engineering Standup - Feb 13
    found_dates = search_dates(first_lines)
    if found_dates:
        for _, parsed_datetime in found_dates:
            return parsed_datetime.date()

    # 3) Fallback to file modified time
    return datetime.fromtimestamp(file_mtime).date()
