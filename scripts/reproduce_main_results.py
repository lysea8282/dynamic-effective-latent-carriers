from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_effective_carriers.reproduction import checklist, run_quick

if __name__ == "__main__":
    print(checklist(run_quick(ROOT)))
