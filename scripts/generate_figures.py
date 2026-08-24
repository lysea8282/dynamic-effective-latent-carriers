from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_effective_carriers.rendering import generate_figures

if __name__ == "__main__":
    for path in generate_figures(ROOT):
        print(path.relative_to(ROOT).as_posix())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "figures" / "joint_extension" / "make_joint_extension_figure.py"),
        ],
        cwd=ROOT,
        check=True,
    )
