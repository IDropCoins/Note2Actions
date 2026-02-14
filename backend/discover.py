from pathlib import Path
from typing import List, Tuple


ALLOWED_EXTENSIONS = {".md", ".txt"}


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
