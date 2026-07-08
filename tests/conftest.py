"""Make the repo root importable (the app is served from the repo root,
not installed as a package — pyproject sets `tool.uv.package = false`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
