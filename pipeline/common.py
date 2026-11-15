"""Shared paths and the artifact gates every stage ends with. Paths come from the environment only."""
import json
import os
import sys
from pathlib import Path


def home() -> Path:
    return Path(os.environ["PIXAL3D_HOME"])


def out() -> Path:
    p = Path(os.environ["PIXAL3D_OUT"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def need_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"pipeline: set {name}")
    return v


def alchemy_env():
    """Settings Alchemy3D reads at import time; call before importing alchemy3d."""
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")  # the SDPA stand-in in pipeline/shims
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("A3D_OFFLOAD", "1")  # denoisers visit the GPU one at a time (a 24 GB card)
    os.environ.setdefault("BLENDER_PATH", str(home() / "blender" / "blender"))
    dino = home() / "dinov3"
    if dino.exists():
        os.environ.setdefault("DINOV3_LOCAL", str(dino))


def record(stage: str, **facts):
    """Write the stage's facts next to its artifacts so a later stage and the tests can read them."""
    f = out() / "stages.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[stage] = facts
    f.write_text(json.dumps(d, indent=1, sort_keys=True))
