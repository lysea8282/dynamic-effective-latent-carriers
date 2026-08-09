from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_effective_carriers.rendering import generate_tables

if __name__ == "__main__":
    for path in generate_tables(ROOT):
        print(path.relative_to(ROOT).as_posix())
