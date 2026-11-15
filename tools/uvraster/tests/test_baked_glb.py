"""Integration checks on a baked GLB (set BAKED_GLB). Controls: a blank texture and shifted UVs must fail."""
import os

import numpy as np
import pytest
import trimesh

path = os.environ.get("BAKED_GLB")
pytestmark = pytest.mark.skipif(not path, reason="BAKED_GLB not set")


def load():
    m = trimesh.load(path, force="mesh", process=False)
    img = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32)
    return m, img


def uv_coverage(m, res):
    uv = m.visual.uv
    tri = uv[m.faces]
    # texel-centre count inside any triangle, cheap bbox-free estimate via area sum
    a = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    return a.sum()


def check_texture_matches_layout(m, img):
    assert m.visual.uv.min() >= -1e-6 and m.visual.uv.max() <= 1 + 1e-6
    covered = uv_coverage(m, img.shape[0])
    assert 0.2 < covered <= 1.0
    # texels under triangles must carry colour: a blank (constant) texture is a failed bake
    assert img.std() > 8.0
    assert (img.reshape(-1, 3).max(1) > 20).mean() > 0.2 * covered


def test_baked_glb_has_a_real_texture():
    m, img = load()
    assert len(m.faces) > 10000
    check_texture_matches_layout(m, img)


def test_control_blank_texture_is_detected():
    m, img = load()
    with pytest.raises(AssertionError):
        check_texture_matches_layout(m, np.full_like(img, 128.0))


def test_control_uv_out_of_range_is_detected():
    m, img = load()
    m.visual.uv = m.visual.uv + 0.5
    with pytest.raises(AssertionError):
        check_texture_matches_layout(m, img)


# ---- content check: an edit keeps its source, so most source occupancy must survive in the edited mesh
source = os.environ.get("SOURCE_GLB")


def occupancy(mesh, lo, hi, n=48, pts=400000):
    pts_ = mesh.sample(pts)
    g = np.clip(((pts_ - lo) / (hi - lo) * n).astype(int), 0, n - 1)
    occ = np.zeros((n, n, n), bool)
    occ[g[:, 0], g[:, 1], g[:, 2]] = True
    return occ


def normalised(src):
    """The generator writes into the -0.5..0.5 cube: centre the source and scale its longest side to 1."""
    lo, hi = src.bounds
    out = src.copy()
    out.vertices = (src.vertices - (lo + hi) / 2) / (hi - lo).max()
    return out


def source_kept(edit, src):
    # both meshes in the same frame after normalising by the source's box (o_voxel output is y-up like the glb)
    lo, hi = src.bounds
    pad = (hi - lo).max() * 0.15
    lo, hi = lo - pad, hi + pad
    s = occupancy(src, lo, hi)
    from scipy import ndimage
    e = ndimage.binary_dilation(occupancy(edit, lo, hi), iterations=1)
    kept = (s & e).sum() / s.sum()
    extra = (occupancy(edit, lo, hi) & ~ndimage.binary_dilation(s, iterations=3)).sum() / occupancy(edit, lo, hi).sum()
    assert kept > 0.8, f"only {kept:.2f} of the source survives"
    assert extra < 0.6, f"{extra:.2f} of the edit lies far from the source"


@pytest.mark.skipif(not source, reason="SOURCE_GLB not set")
def test_edit_keeps_its_source():
    m, _ = load()
    source_kept(m, normalised(trimesh.load(source, force="mesh", process=False)))


@pytest.mark.skipif(not source, reason="SOURCE_GLB not set")
def test_control_a_noise_cube_is_detected():
    s = normalised(trimesh.load(source, force="mesh", process=False))
    lo, hi = s.bounds
    rng = np.random.default_rng(0)
    noise = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
    noise = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(noise.vertices, noise.faces, 0.05), process=False)
    with pytest.raises(AssertionError):
        source_kept(noise, s)
