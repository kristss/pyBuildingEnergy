"""
Pytest configuration for pybuildingenergy tests.

Patches the broken plotly → xarray import chain before the package loads.
This is a workaround for an xarray / packaging version conflict in the local
environment; it does not affect production code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure local src is on sys.path (belt-and-suspenders, also done in __init__.py)
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# Mock the plotly/xarray chain that fails in this environment so the package
# can be imported.  Production code that actually uses plotly (generate_profile)
# will still get a MagicMock and must not be called in unit tests.
for _broken in (
    "xarray",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "plotly.subplots",
    "plotly.io",
):
    if _broken not in sys.modules:
        sys.modules[_broken] = MagicMock()
