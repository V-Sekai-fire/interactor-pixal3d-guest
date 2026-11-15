"""Host side of interactor-remesh-guest: simplify a mesh with remesh.elf (meshoptimizer) run in the godot-sandbox.

REMESH_ELF, GODOT_BIN and GODOT_PROJECT (a Godot project holding addons/godot_sandbox) come from the environment.
The guest keeps vertex positions and returns the reduced index list; avatar_compact then drops unused vertices."""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _env():
    e = dict(os.environ)
    for k in ("REMESH_ELF", "GODOT_BIN", "GODOT_PROJECT"):
        if not e.get(k):
            raise RuntimeError(f"set {k}")
    return e


def _run(args):
    e = _env()
    cmd = [e["GODOT_BIN"], "--path", e["GODOT_PROJECT"], "--script", str(HERE / "remesh.gd"), "--", f"--elf={e['REMESH_ELF']}", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=int(e.get('REMESH_TIMEOUT', '180')))  # Godot can hang after printing its result
    return p.stdout + p.stderr


def gate(planted_um: int) -> str:
    """The guest's own sphere gate; a planted error above its tolerance must report FAIL."""
    for line in _run([f"--gate={planted_um}"]).splitlines():
        if line.startswith("GATE"):
            return line
    raise RuntimeError("remesh gate printed nothing")


def simplify(V, F, target_triangles, error=0.01):
    """Return (V', F') with about target_triangles faces."""
    V = np.asarray(V, dtype="<f4")
    F = np.asarray(F, dtype="<i4")
    with tempfile.TemporaryDirectory(dir=os.environ.get("PIXAL3D_OUT")) as d:
        src, dst = Path(d, "in.bin"), Path(d, "out.bin")
        with open(src, "wb") as f:
            np.array([len(V), F.size], dtype="<i4").tofile(f)
            V.tofile(f)
            F.tofile(f)
        log = _run([f"--in={src}", f"--out={dst}", f"--target={int(target_triangles) * 3}", f"--error={error}"])
        if not dst.exists():
            raise RuntimeError("remesh produced no output:\n" + log[-2000:])
        with open(dst, "rb") as f:
            nv, ni = np.fromfile(f, "<i4", 2)
            remap = np.fromfile(f, "<i4", nv)
            idx = np.fromfile(f, "<i4", ni)
    # remap[i] = old index of new vertex i is what avatar_compact returns for kept vertices
    return V[remap].astype("float64"), idx.reshape(-1, 3)
