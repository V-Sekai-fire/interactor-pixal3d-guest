"""Stage 2: out/source.glb + 1-8 multi-view 2D part maps (PIXAL3D_MAPS, a directory or a comma list) -> per-face labels."""
import glob
import os
import sys

from common import alchemy_env, home, out, record
from gates import gate_labels

o = out()
maps = os.environ.get("PIXAL3D_MAPS")
if not maps:
    record("segment", skipped="PIXAL3D_MAPS not set (Alchemy3D's own 2D map generator is not released)")
    print("segment skipped: no 2D part maps given")
    sys.exit(0)
files = sorted(glob.glob(os.path.join(maps, "*.png"))) if os.path.isdir(maps) else maps.split(",")
if not 1 <= len(files) <= 8:
    sys.exit(f"segment: need 1 to 8 maps, got {len(files)}")

os.environ.setdefault("HF_HOME", str(home() / "hf"))
alchemy_env()
import numpy as np
import trimesh

from alchemy3d.pipelines import Alchemy3DSegmentPipeline

pipe = Alchemy3DSegmentPipeline.from_pretrained(model_name_or_path="libadi/Alchemy3D-Segment")
pipe.cuda()
_, preds = pipe.run(source=str(o / "source.glb"), image=[files], return_label=True, comparison_video=None)
labels = np.asarray(preds[0])
np.save(o / "labels.npy", labels)
m = trimesh.load(o / "source.glb", force="mesh", process=False)
problems = gate_labels(labels, len(m.faces))
if problems:
    sys.exit("segment gate failed: " + "; ".join(problems))
palette = (np.array([[(i * 67) % 256, (i * 131) % 256, (i * 29 + 90) % 256, 255] for i in range(labels.max() + 1)]))
trimesh.Trimesh(m.vertices, m.faces, face_colors=palette[labels].astype("uint8"), process=False).export(o / "segments.ply")
record("segment", maps=len(files), parts=int(len(np.unique(labels))))
print("segment ok:", len(np.unique(labels)), "parts")
