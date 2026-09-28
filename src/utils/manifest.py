from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyarrow",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    status = _run("status", "--porcelain")
    return {
        "repo": str(repo),
        "commit": _run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_total_mem_bytes"] = int(props.total_memory)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = None
    return info


def build_manifest(**sections: Any) -> Dict[str, Any]:
    """Base run manifest plus caller-provided sections (model, data, config, ...)."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware_info(),
        **sections,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def make_run_dir(parent: Path, tag: str) -> Path:
    """Create a fresh, never-reused output directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = parent / f"{stamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
