"""Stage 4: out/edit.pkl -> out/edited.glb (UV unwrap and texture bake through o_voxel with the torch UV rasteriser)."""
import pickle
import sys

import numpy as np
import torch

from common import alchemy_env, out, record
from gates import gate_source_kept, gate_texture

alchemy_env()
import remesh
import o_voxel

o = out()
with open(o / "edit.pkl", "rb") as f:
    m = pickle.load(f)
V = m.vertices.detach().cpu().numpy().astype("float64")
F = m.faces.detach().cpu().numpy().astype("int32")
if len(F) > 3_000_000:  # a 24 GB card cannot simplify 37 M faces; attributes come from the voxel volume anyway
    V, F = remesh.simplify(V, F, target_triangles=2_000_000)  # remesh.elf (meshoptimizer), not a second implementation
    V, F = V.astype("float32"), F.astype("int32")
    m.vertices = torch.from_numpy(V).float().cuda()
    m.faces = torch.from_numpy(F).int().cuda()
glb = o_voxel.postprocess.to_glb(
    vertices=m.vertices, faces=m.faces, attr_volume=m.attrs, coords=m.coords, attr_layout=m.layout,
    voxel_size=m.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=300000,
    texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
glb.export(o / "edited.glb")
problems = gate_texture(o / "edited.glb") + gate_source_kept(o / "edited.glb", o / "source.glb")
if problems:
    sys.exit("bake gate failed: " + "; ".join(problems))
record("bake", faces=int(len(glb.faces)))
print("bake ok:", o / "edited.glb")
