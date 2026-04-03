"""
Shared repository path setup for test/*.py scripts.

Run tests from the repo root, e.g.:

    python3 test/test_drone_link.py
    python3 test/test_drone_link.py --connection udpin:127.0.0.1:14552
"""
import os
import sys


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def prepend_import_paths():
    """Insert repo root, modules/, and cv/models/ so imports match mission_mvp."""
    root = repo_root()
    for p in (root, os.path.join(root, "modules"), os.path.join(root, "cv", "models")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
