"""Artifact gates. Each returns a list of problems (empty = pass); the stage scripts exit non-zero on any."""
import numpy as np
import trimesh


def gate_mesh(path, min_faces=1000, lo=None, hi=None):
    problems = []
    m = trimesh.load(path, force="mesh", process=False)
    if len(m.faces) < min_faces:
        problems.append(f"{path}: only {len(m.faces)} faces")
    if not np.isfinite(m.vertices).all():
        problems.append(f"{path}: non-finite vertices")
    ext = m.bounds[1] - m.bounds[0]
    if ext.max() <= 0:
        problems.append(f"{path}: zero extent")
    return problems


def gate_labels(labels, n_faces, min_parts=2, max_background=0.95):
    problems = []
    labels = np.asarray(labels)
    if labels.shape != (n_faces,):
        problems.append(f"labels shape {labels.shape} != ({n_faces},)")
        return problems
    u, c = np.unique(labels, return_counts=True)
    if len(u) < min_parts:
        problems.append(f"only {len(u)} distinct label(s)")
    if c.max() / n_faces > max_background:
        problems.append(f"one label covers {c.max() / n_faces:.2f} of the faces")
    return problems


def gate_texture(glb, min_std=8.0):
    problems = []
    m = trimesh.load(glb, force="mesh", process=False)
    uv = m.visual.uv
    if uv.min() < -1e-6 or uv.max() > 1 + 1e-6:
        problems.append("UVs outside 0..1")
    img = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32)
    if img.std() < min_std:
        problems.append(f"texture is flat (std {img.std():.1f})")
    return problems


def gate_source_kept(edit_glb, source_glb, kept_min=0.8, extra_max=0.6):
    """An edit keeps its source: most source occupancy must survive and the rest must stay near it."""
    from scipy import ndimage

    src = trimesh.load(source_glb, force="mesh", process=False)
    lo, hi = src.bounds
    src = src.copy()
    src.vertices = (src.vertices - (lo + hi) / 2) / (hi - lo).max()
    e = trimesh.load(edit_glb, force="mesh", process=False)
    lo, hi = src.bounds
    pad = (hi - lo).max() * 0.15
    lo, hi = lo - pad, hi + pad

    def occ(m, n=48, pts=400000):
        p = m.sample(pts)
        g = np.clip(((p - lo) / (hi - lo) * n).astype(int), 0, n - 1)
        o = np.zeros((n, n, n), bool)
        o[g[:, 0], g[:, 1], g[:, 2]] = True
        return o

    s, eo = occ(src), occ(e)
    kept = (s & ndimage.binary_dilation(eo, iterations=1)).sum() / s.sum()
    extra = (eo & ~ndimage.binary_dilation(s, iterations=3)).sum() / eo.sum()
    problems = []
    if kept < kept_min:
        problems.append(f"only {kept:.2f} of the source survives the edit")
    if extra > extra_max:
        problems.append(f"{extra:.2f} of the edit lies far from the source")
    return problems
