from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_effective_carriers.utils import read_json

if __name__ == "__main__":
    print(json.dumps(read_json(ROOT / "results/expected/paper_summary.json")["temporal_stability"], indent=2))
