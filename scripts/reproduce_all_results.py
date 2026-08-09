from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_effective_carriers.reproduction import checklist, run_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every packaged result and optionally retrain models.")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run_full(ROOT, retrain=args.retrain, seeds=args.seeds, device=args.device)
    print(checklist(result["quick"]))
    print(json.dumps({"validated_carrier_artifact_count": result["validated_carrier_artifact_count"], "retrained": len(result["retraining"])}, indent=2))


if __name__ == "__main__":
    main()
