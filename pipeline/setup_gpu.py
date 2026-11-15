"""One-time GPU setup: CUDA wheels without their git dependency, the Alchemy3D checkout with our patch, Blender."""
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from common import home

WHEELS = "https://huggingface.co/ifire/pixal3d-cuda-wheels/resolve/main"
PACKAGES = ["flex_gemm-1.0.0", "cumesh-0.0.1", "o_voxel-0.0.1"]
ALCHEMY = "https://github.com/libd1/Alchemy3D.git"
BLENDER = "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz"
root = Path(os.environ["PIXI_PROJECT_ROOT"])


def run(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


h = home()
h.mkdir(parents=True, exist_ok=True)
# o_voxel's metadata asks for cumesh from git; the wheel we ship sits beside it, so install without dependencies
run("uv", "pip", "install", "--python", sys.executable, "--no-deps", *[f"{WHEELS}/{p}-cp312-cp312-linux_x86_64.whl" for p in PACKAGES])

a = h / "Alchemy3D"
if not a.exists():
    run("git", "clone", "--depth", "50", ALCHEMY, a)
patch = root / "pipeline" / "alchemy3d.patch"
if subprocess.run(["git", "-C", a, "apply", "--check", "-R", patch], capture_output=True).returncode == 0:
    print("Alchemy3D patch already applied")
else:
    run("git", "-C", a, "apply", patch)

b = h / "blender"
if not (b / "blender").exists():
    tar = h / "blender.tar.xz"
    urllib.request.urlretrieve(BLENDER, tar)
    with tarfile.open(tar) as t:
        t.extractall(h)
    (h / "blender-4.3.2-linux-x64").rename(b)
    tar.unlink()
    py = next((b / "4.3" / "python" / "bin").glob("python3*"))
    run(py, "-m", "ensurepip", "--upgrade")
    run(py, "-m", "pip", "install", "Pillow")
print("setup done")
