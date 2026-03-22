"""Pytest configuration for tests."""

import sys
from pathlib import Path

# Add src to path so 'from failure_event import FailureEvent' works
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
