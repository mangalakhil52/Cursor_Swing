#!/usr/bin/env python3
"""Back-compat wrapper — prefer scripts/refresh_universe.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("refresh_universe.py")
    sys.argv = [str(script), "--source", "nifty500", "-o", "data/nifty500.csv"]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
