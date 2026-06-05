"""
Pytest configuration for pybuildingenergy tests.

NOTE FOR UPSTREAM PR: This file contains a local environment workaround.
Remove or replace it before opening an upstream PR. The xarray/packaging
version conflict it patches does not exist in a clean environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src is on sys.path (belt-and-suspenders, also done in __init__.py)
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# Guard: only mock xarray/plotly if the actual import chain is broken.
# This detects the specific packaging.version.Version bug that occurs when
# numpy's __version__ is None in certain conda environments.
_needs_mock = False
try:
    import xarray  # noqa: F401
except (TypeError, AttributeError):
    _needs_mock = True
except ImportError:
    pass  # genuine missing package — don't mask it

if _needs_mock:
    from unittest.mock import MagicMock
    for _mod in (
        "xarray",
        "plotly",
        "plotly.express",
        "plotly.graph_objects",
        "plotly.subplots",
        "plotly.io",
    ):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()
