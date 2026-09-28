import sys
from pathlib import Path

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch-free test runs
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
