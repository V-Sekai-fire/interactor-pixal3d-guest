"""Every gate passes a good artifact and fails its planted control."""
import os

import numpy as np
import pytest
import trimesh
from PIL import Image

from gates import gate_labels, gate_mesh, gate_source_kept, gate_texture


def sphere(path, centre=(0, 0, 0), radius=1.0, sub=4, texture=None):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=radius)
    m.apply_translation(centre)
    if texture is not None:
        uv = np.c_[(np.arctan2(m.vertices[:, 1], m.vertices[:, 0]) / (2 * np.pi) + 0.5), m.vertices[:, 2] / 2 + 0.5]
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture)))
    m.export(path)
    return path


def test_gate_mesh(tmp_path):
    assert gate_mesh(sphere(tmp_path / "a.glb")) == []
    assert gate_mesh(sphere(tmp_path / "b.glb", sub=1)) != []  # too few faces


def test_gate_labels():
    ok = np.arange(5000) % 5
    assert gate_labels(ok, 5000) == []
    assert gate_labels(np.zeros(5000, int), 5000) != []  # one label
    assert gate_labels(ok[:100], 5000) != []  # wrong length
    skew = np.zeros(5000, int)
    skew[:10] = 1
    assert gate_labels(skew, 5000) != []  # background swallows the mesh


def test_gate_texture(tmp_path):
    rng = np.random.default_rng(0)
    assert gate_texture(sphere(tmp_path / "t.glb", texture=rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))) == []
    assert gate_texture(sphere(tmp_path / "flat.glb", texture=np.full((64, 64, 3), 128, np.uint8))) != []


def test_gate_source_kept(tmp_path):
    # the gate reads the edit in the unit frame the generator writes (longest side 1), so the spheres have radius 0.5
    src = sphere(tmp_path / "src.glb", radius=0.5)
    assert gate_source_kept(sphere(tmp_path / "same.glb", radius=0.5), src) == []
    # a slightly larger shell around the source keeps it: an edit may add material
    assert gate_source_kept(sphere(tmp_path / "bump.glb", radius=0.52), src) == []
    # control: a small sphere somewhere else does not keep it
    away = sphere(tmp_path / "away.glb", centre=(0.45, 0, 0), radius=0.2)
    assert gate_source_kept(away, src) != []
    # control: a noise cube around it does not keep it either
    cube = trimesh.creation.box(extents=(1.6, 1.6, 1.6))
    cube = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(cube.vertices, cube.faces, 0.05), process=False)
    cube.export(tmp_path / "cube.glb")
    assert gate_source_kept(tmp_path / "cube.glb", src) != []


REMESH = pytest.mark.skipif(not all(os.environ.get(k) for k in ("REMESH_ELF", "GODOT_BIN", "GODOT_PROJECT")),
                            reason="REMESH_ELF, GODOT_BIN and GODOT_PROJECT not set")


@REMESH
def test_remesh_guest_gate_and_its_planted_control():
    import remesh

    assert "PASS" in remesh.gate(0)
    assert "FAIL" in remesh.gate(2000)


@REMESH
def test_remesh_simplify_keeps_the_sphere():
    import remesh

    m = trimesh.creation.icosphere(subdivisions=5)
    V, F = remesh.simplify(m.vertices, m.faces, target_triangles=2000)
    assert 500 < len(F) <= 2400
    assert F.max() < len(V) and F.min() >= 0
    assert np.abs(np.linalg.norm(V[np.unique(F)], axis=1) - 1).max() < 0.05
    # control: asking for one triangle does not return the sphere
    V1, F1 = remesh.simplify(m.vertices, m.faces, target_triangles=4, error=1.0)
    assert len(F1) < 100
