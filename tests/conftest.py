"""
Makes src/<subfolder> modules importable in tests, e.g.
`from validate_schema import validate` instead of needing package-style
`from src.data.validate_schema import validate`.

This matches how the scripts themselves import (see src/data/ingest.py's
`from validate_schema import validate`) -- both rely on their own
directory being on sys.path, which Python does automatically when a
script is run directly, but pytest needs this file to do the same.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"

for subdir in SRC_DIR.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("__"):
        sys.path.insert(0, str(subdir))