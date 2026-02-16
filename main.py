"""Root entrypoint for `uvicorn main:app --reload`."""

import importlib
import sys
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parent / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

_api = importlib.import_module("api")
app = _api.app

__all__ = ["app"]
