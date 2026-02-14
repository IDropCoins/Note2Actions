from pathlib import Path


def load_text(file_path: str) -> str:
    """
    Safely load a text file and normalize formatting.

    - Reads as UTF-8 with fallback replacement
    - Normalizes Windows/Mac newlines to \n
    - Strips trailing whitespace on each line
    - Preserves paragraph structure
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read safely
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Normalize newlines
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]

    # Reassemble
    normalized = "\n".join(lines).strip()

    return normalized
