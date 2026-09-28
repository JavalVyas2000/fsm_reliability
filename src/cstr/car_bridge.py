"""
Import the ctrl-alt-recover CSTR case study WITHOUT copying or modifying it.

cstr_case.py imports LangGraph/LangChain at module level for its agent graph and LLM
clients. This project uses only its simulator, monitoring, prompt construction and
rollout validator, so light stand-ins are registered for those modules when they are not
installed. The stand-ins raise if anything tries to use them, so no hidden LLM or graph
behaviour can leak in.

cstr_case.py also rebinds sys.stdout at import; the original stream is restored after the
import.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict

CAR_ROOT = Path("C:/Users/jv624/Desktop/ctrl-alt-recover")
CSTR_DIR = CAR_ROOT / "case_studies" / "cstr_case"

_MODULE = None


def ensure_paths() -> None:
    """Make the CAR simulator modules importable (needed e.g. to unpickle snapshots)."""
    for p in (str(CSTR_DIR), str(CSTR_DIR.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_paths()


class _Unavailable:
    """Stand-in for an unused LangChain/LangGraph object; any use is an error."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("LangChain/LangGraph stand-in used; this project must not call the CAR agent graph or LLM clients")


def _stub(name: str, **attrs: Any) -> None:
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mod.__car_bridge_stub__ = True
    sys.modules[name] = mod


def load_cstr_case():
    """Import and return the cstr_case module (cached)."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _stub("langgraph")
    _stub("langgraph.graph", StateGraph=_Unavailable, START="__start__", END="__end__")
    _stub("langchain")
    _stub("langchain.callbacks", get_openai_callback=_Unavailable)
    _stub("langchain_openai", ChatOpenAI=_Unavailable)
    _stub("langchain_ollama", ChatOllama=_Unavailable)
    for p in (str(CSTR_DIR), str(CSTR_DIR.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)

    global _CAR_STDOUT
    saved_stdout = sys.stdout
    try:
        sys.stdout = sys.__stdout__
        _MODULE = importlib.import_module("cstr_case")
    finally:
        # cstr_case opened a second writer on fd 1; keep it referenced so garbage
        # collection never closes the process's stdout descriptor.
        _CAR_STDOUT = sys.stdout
        sys.stdout = saved_stdout
    return _MODULE


_CAR_STDOUT = None


def car_provenance() -> Dict[str, Any]:
    """Commit and dirty state of ctrl-alt-recover plus sha256 of the files used."""
    from src.utils.manifest import sha256_file

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=CAR_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    files = ["cstr_case.py", "cstr_digital_twin.py", "cstr_anomaly_threshold_test.py", "graph_retrieval_code.py"]
    return {
        "repo": str(CAR_ROOT),
        "commit": git("rev-parse", "HEAD"),
        "dirty_files": (git("status", "--porcelain") or "").splitlines(),
        "sha256": {f: sha256_file(CSTR_DIR / f) for f in files},
    }
