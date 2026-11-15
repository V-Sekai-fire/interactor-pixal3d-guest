import sys, numpy as np, trimesh, mitsuba as mi
from PIL import Image
mi.set_variant("cuda_ad_rgb")
src, out = sys.argv[1], sys.argv[2]
m = trimesh.load(src, force="mesh", process=False)
tex = m.visual.material.baseColorTexture.convert("RGB"); tex.save("/tmp/_tex.png")
v = np.asarray(m.vertices); c = (v.max(0) + v.min(0)) / 2; s = (v.max(0) - v.min(0)).max()
m.vertices = (v - c) / s
m.visual.uv = np.c_[m.visual.uv[:, 0], 1 - m.visual.uv[:, 1]]  # mitsuba bitmap origin is top-left
m.export("/tmp/_m.ply")
import os
def view(az):
    t = mi.ScalarTransform4f
    return mi.load_dict({"type": "scene", "integrator": {"type": "path"},
        "sensor": {"type": "perspective", "fov": 35, "to_world": t().look_at(origin=[2.2 * np.sin(az), 0.6, 2.2 * np.cos(az)], target=[0, 0, 0], up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": 480, "height": 480}, "sampler": {"type": "independent", "sample_count": 48}},
        "light": {"type": "constant", "radiance": {"type": "rgb", "value": 0.9}},
        "key": {"type": "point", "position": [2, 3, 2], "intensity": {"type": "rgb", "value": 18}},
        "mesh": {"type": "ply", "filename": "/tmp/_m.ply", "face_normals": False,
                 "bsdf": {"type": "diffuse", "reflectance": {"type": "bitmap", "filename": "/tmp/_tex.png"}}}})
ims = [np.clip(np.array(mi.render(view(a))) ** (1 / 2.2), 0, 1) for a in (0, 1.57, 3.14, 4.71)]
Image.fromarray((np.concatenate(ims, 1) * 255).astype("uint8")).save(out)
print("saved", out)
