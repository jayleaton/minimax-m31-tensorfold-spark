"""CPU-only tests of the thermal / memory guard (scripts/guard): durable policy, admission and loopback transport.

    tests/run.sh   (or: python -m pytest -q tests/guard)

No GPU, Docker, SSH or systemd is touched: the backend's host actions are replaced by local fakes.

Ported from tests/guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE.parents[1] / "scripts" / "guard"
for path in (GUARD, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
