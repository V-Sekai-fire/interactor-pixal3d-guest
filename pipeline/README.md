# Pipeline notes

## Shared fedora setup under ~/og2 (another session's, not ours)

Our runs use `~/scratch/pg-pipe` and `~/scratch/a3d`. The Poseswap/AUX-label session keeps a second Alchemy3D
setup in `~/og2` (a3d-env, ext, pixal3d/hf). Before pointing a run of ours at it, know that:

- `~/og2/pixal3d/hf/hub` holds two **aliased weights**, symlinks and not the real repos:
  - `facebook/dinov3-vitl16-pretrain-lvd1689m` -> the camenduru copy (the official one is gated; same model).
  - `briaai/RMBG-2.0` -> `ZhengPeng7/BiRefNet_HR-matting`. **Different weights**, and the RMBG licence was not checked.
    Any matte or crop made through that alias differs from the real RMBG-2.0, so do not compare it with other runs.
- `~/og2/a3d-env/run.sh` sets `SPARSE_ATTN_BACKEND=xformers` (xformers 0.0.29.post3). Our runs set `flash_attn`,
  which resolves to the SDPA shim in `pipeline/shims`.
- Segment there runs `HF_HUB_OFFLINE=1`, without Blender (trimesh plus O-Voxel wrapper `~/og2/seg_nb.py`), and its
  `return_label` gives anonymous MeanShift cluster ids, not part classes.
- Ask that session before changing any of it (claims board, 2026-11-15 04:56).
