"""Tests for the UV-space rasterizer stand-in. Each check has a planted-defect control that must fail it.

  PYTHONPATH=tools/uvraster python -m pytest tools/uvraster/tests -q
"""
import importlib
import math

import numpy as np
import pytest
import torch

import nvdiffrast.torch as dr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 64


def clip(uv):
    uv = torch.as_tensor(uv, dtype=torch.float32, device=DEV)
    return torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], 1)[None]


def raster(uv, tri, res=RES, fn=None):
    fn = fn or dr.rasterize
    t = torch.as_tensor(tri, dtype=torch.int32, device=DEV)
    r, _ = fn(dr.RasterizeCudaContext(), clip(uv), t, resolution=[res, res])
    return r, t


def reference(uv, tri, res=RES):
    """Brute force: every texel centre against every triangle, in float64."""
    uv = np.asarray(uv, dtype=np.float64)
    out = np.zeros((res, res, 3))  # u, v, id
    ys, xs = np.mgrid[0:res, 0:res]
    px = (xs + 0.5) / res
    py = (ys + 0.5) / res
    for i, (a, b, c) in enumerate(tri):
        A, B, C = uv[a], uv[b], uv[c]
        det = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        l0 = ((B[1] - C[1]) * (px - C[0]) + (C[0] - B[0]) * (py - C[1])) / det
        l1 = ((C[1] - A[1]) * (px - C[0]) + (A[0] - C[0]) * (py - C[1])) / det
        l2 = 1 - l0 - l1
        inside = (l0 > 1e-6) & (l1 > 1e-6) & (l2 > 1e-6)
        out[inside] = np.stack([l1[inside], l2[inside], np.full(inside.sum(), i + 1)], 1)
    return out


def test_single_triangle_area():
    r, _ = raster([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], [[0, 1, 2]], res=256)
    covered = (r[0, ..., 3] > 0).float().mean().item()
    assert covered == pytest.approx(0.5 * 0.8 * 0.8, abs=0.01)


def test_matches_brute_force_on_random_layout():
    rng = np.random.default_rng(1)
    # a grid of small separated triangles, random winding
    uv, tri = [], []
    for gy in range(6):
        for gx in range(6):
            o = np.array([gx, gy]) / 6 + 0.01
            p = o + rng.uniform(0, 0.14, (3, 2))
            tri.append([len(uv), len(uv) + 1, len(uv) + 2])
            uv.extend(p.tolist())
    r, _ = raster(uv, tri)
    ref = reference(uv, tri)
    got = r[0].cpu().numpy()
    both = (got[..., 3] > 0) & (ref[..., 2] > 0)
    # texels strictly inside agree on id and barycentrics; edge texels may differ by inclusion rule only
    assert both.sum() > 100
    assert (got[both][:, 3] == ref[both][:, 2]).all()
    assert np.abs(got[both][:, :2] - ref[both][:, :2]).max() < 1e-4
    only_got = (got[..., 3] > 0) & ~(ref[..., 2] > 0)
    only_ref = ~(got[..., 3] > 0) & (ref[..., 2] > 0)
    assert only_got.sum() + only_ref.sum() <= 0.02 * both.sum()


def test_interpolation_is_exact_for_an_affine_attribute():
    uv = np.array([[0.1, 0.2], [0.8, 0.15], [0.3, 0.9]])
    r, t = raster(uv, [[0, 1, 2]])
    # attribute = 3 u + 5 v + 1 is affine in the layout, so the interpolation must reproduce it at every texel
    attr = torch.as_tensor(3 * uv[:, 0] + 5 * uv[:, 1] + 1, dtype=torch.float32, device=DEV)[None, :, None]
    got, _ = dr.interpolate(attr, r, t)
    ys, xs = np.mgrid[0:RES, 0:RES]
    want = 3 * (xs + 0.5) / RES + 5 * (ys + 0.5) / RES + 1
    m = (r[0, ..., 3] > 0).cpu().numpy()
    assert m.sum() > 500
    assert np.abs(got[0, ..., 0].cpu().numpy()[m] - want[m]).max() < 1e-4
    assert (got[0, ..., 0].cpu().numpy()[~m] == 0).all()


def test_face_ids_follow_the_input_order_across_chunks(monkeypatch):
    # many triangles force several candidate chunks
    n = 40
    uv, tri = [], []
    for k in range(n * n):
        gx, gy = k % n, k // n
        o = np.array([gx, gy]) / n
        s = 0.9 / n
        uv.extend([(o + [0.05 * s, 0.05 * s]).tolist(), (o + [s * 0.9, 0.05 * s]).tolist(), (o + [0.05 * s, s * 0.9]).tolist()])
        tri.append([3 * k, 3 * k + 1, 3 * k + 2])
    r, _ = raster(uv, tri, res=512)
    ids = r[0, ..., 3].long().cpu().numpy()
    ys, xs = np.mgrid[0:512, 0:512]
    cell = np.floor((ys + 0.5) / 512 * n).astype(int) * n + np.floor((xs + 0.5) / 512 * n).astype(int)
    inside = ids > 0
    assert inside.sum() > 5000
    assert (ids[inside] - 1 == cell[inside]).all()


def test_degenerate_triangle_is_ignored():
    r, _ = raster([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], [[0, 1, 2]])
    assert (r[0, ..., 3] == 0).all()


# ---- controls: the checks above must fail on a planted defect
def swapped_uv(ctx, pos, tri, resolution, **k):
    r, _ = dr.rasterize(ctx, pos, tri, resolution)
    return r[..., [1, 0, 2, 3]], None


def shifted_id(ctx, pos, tri, resolution, **k):
    r, _ = dr.rasterize(ctx, pos, tri, resolution)
    r = r.clone()
    r[..., 3] = torch.where(r[..., 3] > 0, r[..., 3] + 1, r[..., 3])
    return r, None


def half_pixel(ctx, pos, tri, resolution, **k):
    p = pos.clone()
    p[..., 0] += 1.0 / resolution[1]  # a half texel shift in NDC
    return dr.rasterize(ctx, p, tri, resolution)


@pytest.mark.parametrize("bad", [swapped_uv, shifted_id, half_pixel])
def test_controls_are_detected(bad):
    uv = np.array([[0.1, 0.2], [0.8, 0.15], [0.3, 0.9]])
    r, t = raster(uv, [[0, 1, 2], [0, 0, 0]], fn=bad)  # a second (degenerate) face keeps shifted ids in range
    attr = torch.as_tensor(3 * uv[:, 0] + 5 * uv[:, 1] + 1, dtype=torch.float32, device=DEV)[None, :, None]
    got, _ = dr.interpolate(attr, r, t)
    ys, xs = np.mgrid[0:RES, 0:RES]
    want = 3 * (xs + 0.5) / RES + 5 * (ys + 0.5) / RES + 1
    m = (r[0, ..., 3] > 0).cpu().numpy()
    ref = reference(uv, [[0, 1, 2]])
    ids_ok = (r[0, ..., 3].cpu().numpy()[m] == 1).all()
    err = np.abs(got[0, ..., 0].cpu().numpy()[m] - want[m]).max() if m.any() else 0
    cover_ok = abs(m.sum() - (ref[..., 2] > 0).sum()) <= 0.02 * (ref[..., 2] > 0).sum()
    assert not (ids_ok and err < 1e-4 and cover_ok), "planted defect was not detected"
