from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_effective_carriers.artifacts import verify_artifacts

if __name__ == "__main__":
    result = verify_artifacts(ROOT)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
