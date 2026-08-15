from __future__ import annotations

import json

from joint_velocity import run_one


if __name__ == "__main__":
    print(json.dumps(run_one("operator_extension_summary.json"), indent=2))
