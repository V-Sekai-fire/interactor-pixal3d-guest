# uvraster

`nvdiffrast/torch/__init__.py`: a UV-space rasterizer (`RasterizeCudaContext`, `rasterize`, `interpolate`) in plain torch, enough for `o_voxel.postprocess.to_glb`'s texture bake. Not a general rasterizer: no depth test, no perspective, no gradients. MIT, no NVIDIA code.

    PYTHONPATH=tools/uvraster python -m pytest tools/uvraster/tests -q          # 8 unit tests, 3 planted-defect controls
    BAKED_GLB=out.glb PYTHONPATH=tools/uvraster python -m pytest tools/uvraster/tests -q   # plus 3 checks on a baked GLB

`bake_glb.py <pickle> <out.glb>` runs `to_glb` (pre-clusters meshes over 3 M faces so a 24 GB card can simplify them). `render_mitsuba.py <glb> <png>` renders four views with Mitsuba 3 (cuda_ad_rgb).
