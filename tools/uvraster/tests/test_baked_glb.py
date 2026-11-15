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
