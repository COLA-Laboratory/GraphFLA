"""Explicit, working-directory-independent entry point for literature tests."""

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    raise SystemExit(pytest.main(["-c", str(HERE.parent / "pytest.ini"),
                                 str(HERE), *sys.argv[1:]]))
