from __future__ import annotations

import json

from joint_velocity import reproduce_all_summaries


if __name__ == "__main__":
    print(json.dumps(reproduce_all_summaries(), indent=2))
