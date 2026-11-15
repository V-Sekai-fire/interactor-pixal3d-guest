"""Stage 1: view image(s) -> out/source.glb with Pixal3D multi-view.

PIXAL3D_IMAGE is the front view (az 0, el 0); PIXAL3D_VIEWS adds orbit views as "az,el:path;az,el:path" (up to 3).
PIXAL3D_SOURCE_GLB skips generation and uses that mesh. The Space's own helpers (camera rig, matting, shared
crop) are taken from its app.py by parsing it, so this stage cannot drift from the Space."""
import ast
import math
import os
import shutil
import sys

from common import alchemy_env, home, need_env, out, record
from gates import gate_mesh

o = out()
src = os.environ.get("PIXAL3D_SOURCE_GLB")
if src:
    shutil.copy(src, o / "source.glb")
    record("generate", mode="given", source=src)
else:
    os.environ.setdefault("HF_HOME", str(home() / "hf"))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    import numpy as np
    import torch
    from PIL import Image

    import o_voxel
    from trellis2.pipelines import Pixal3DMVImageTo3DPipeline

    space = home() / "space"
    tree = ast.parse((space / "app.py").read_text())
    want_fn = {"_matte", "_shared_square_crop", "_cond_tensor", "_c2w", "build_views", "distance_from_fov",
               "compute_f_pixels", "build_image_cond_model"}
    want_val = {"MV_CAMERA_ANGLE_X", "MV_DISTANCE", "MV_MESH_SCALE", "IMAGE_COND_CONFIGS", "CASCADE_MAX_NUM_TOKENS"}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want_fn)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want_val for t in n.targets))]
    ns = {"torch": torch, "np": np, "Image": Image, "math": math, "__name__": "space_helpers"}
    exec("from typing import *", ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "app.py", "exec"), ns)

    pipeline = Pixal3DMVImageTo3DPipeline.from_pretrained(str(home() / "weights"), "pipeline_mv.json")
    for attr, key in (("image_cond_model_ss", "ss"), ("image_cond_model_shape_512", "shape_512"),
                      ("image_cond_model_shape_1024", "shape_1024"), ("image_cond_model_tex_1024", "tex_1024")):
        setattr(pipeline, attr, ns["build_image_cond_model"](ns["IMAGE_COND_CONFIGS"][key]))
    pipeline.low_vram = True  # each stage visits the GPU in turn (24 GB card)
    pipeline.cuda()
    ns["pipeline"] = pipeline

    views = [((0.0, 0.0), Image.open(need_env("PIXAL3D_IMAGE")))]
    for item in filter(None, os.environ.get("PIXAL3D_VIEWS", "").split(";")):
        pose, path = item.split(":", 1)
        az, el = (float(x) for x in pose.split(","))
        views.append(((az, el), Image.open(path)))
    seed = int(os.environ.get("PIXAL3D_SEED", "42"))
    torch.manual_seed(seed)
    built = ns["build_views"]([im for _, im in views], [p for p, _ in views], ns["MV_CAMERA_ANGLE_X"], ns["MV_DISTANCE"])
    _, (shape_slat, tex_slat, res) = pipeline.run_mv(
        built, seed=seed, return_latent=True, pipeline_type=f"{os.environ.get('PIXAL3D_RESOLUTION', '1024')}_cascade",
        max_num_tokens=ns["CASCADE_MAX_NUM_TOKENS"])
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(os.environ.get("PIXAL3D_DECIMATE", "300000")), texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
    glb.export(o / "source.glb")
    record("generate", mode="pixal3d-mv", views=len(views), seed=seed)

problems = gate_mesh(o / "source.glb")
if problems:
    sys.exit("generate gate failed: " + "; ".join(problems))
print("generate ok:", o / "source.glb")
