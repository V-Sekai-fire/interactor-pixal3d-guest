"""Stage 5: out/edited.glb -> out/preview.png, four Mitsuba 3 views (CUDA when the driver offers it, CPU otherwise)."""
import numpy as np
import trimesh
import mitsuba as mi
from PIL import Image

from common import out, record

o = out()
try:
    mi.set_variant("cuda_ad_rgb")
    mi.render(mi.load_dict(mi.cornell_box()), spp=1)
except Exception:
    mi.set_variant("scalar_rgb")
m = trimesh.load(o / "edited.glb", force="mesh", process=False)
v = np.asarray(m.vertices)
m.vertices = (v - (v.max(0) + v.min(0)) / 2) / (v.max(0) - v.min(0)).max()
m.visual.material.baseColorTexture.convert("RGB").save(o / "_texture.png")
m.visual.uv = np.c_[m.visual.uv[:, 0], 1 - m.visual.uv[:, 1]]  # Mitsuba bitmaps start at the top
m.export(o / "_mesh.ply")


def view(az):
    t = mi.ScalarTransform4f
    return mi.load_dict({
        "type": "scene", "integrator": {"type": "path"},
        "sensor": {"type": "perspective", "fov": 35,
                   "to_world": t().look_at(origin=[2.2 * np.sin(az), 0.6, 2.2 * np.cos(az)], target=[0, 0, 0], up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": 480, "height": 480}, "sampler": {"type": "independent", "sample_count": 48}},
        "light": {"type": "constant", "radiance": {"type": "rgb", "value": 0.9}},
        "key": {"type": "point", "position": [2, 3, 2], "intensity": {"type": "rgb", "value": 18}},
        "mesh": {"type": "ply", "filename": str(o / "_mesh.ply"),
                 "bsdf": {"type": "diffuse", "reflectance": {"type": "bitmap", "filename": str(o / "_texture.png")}}}})


ims = [np.clip(np.array(mi.render(view(a))) ** (1 / 2.2), 0, 1) for a in (0, 1.57, 3.14, 4.71)]
Image.fromarray((np.concatenate(ims, 1) * 255).astype("uint8")).save(o / "preview.png")
record("render", variant=mi.variant())
print("render ok:", o / "preview.png")
