"""UV-space-only stand-in for nvdiffrast.torch (RasterizeCudaContext, rasterize, interpolate).
Enough for o_voxel.postprocess.to_glb, which rasterizes the UV layout (z=0, w=1) once. MIT, no NVIDIA code."""
import torch


class RasterizeCudaContext:
    def __init__(self, *a, **k):
        pass


def rasterize(ctx, pos, tri, resolution, grad_db=False, **k):
    # pos [1,V,4] clip space, tri [T,3] int32. Output [1,H,W,4] = (u, v, z, tri_id + 1), row py at y_ndc = (py + 0.5) / H * 2 - 1.
    H, W = int(resolution[0]), int(resolution[1])
    dev = pos.device
    p = pos[0, :, :2].float()
    px_ = (p[:, 0] * 0.5 + 0.5) * W
    py_ = (p[:, 1] * 0.5 + 0.5) * H
    tri = tri.long()
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ax, ay, bx, by, cx, cy = px_[a], py_[a], px_[b], py_[b], px_[c], py_[c]
    x0 = torch.floor(torch.minimum(torch.minimum(ax, bx), cx) - 0.5).clamp(0, W - 1).long()
    x1 = torch.ceil(torch.maximum(torch.maximum(ax, bx), cx) - 0.5).clamp(0, W - 1).long()
    y0 = torch.floor(torch.minimum(torch.minimum(ay, by), cy) - 0.5).clamp(0, H - 1).long()
    y1 = torch.ceil(torch.maximum(torch.maximum(ay, by), cy) - 0.5).clamp(0, H - 1).long()
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    cnt = bw * bh
    out = torch.zeros(H * W, 4, device=dev)
    # expand candidates in chunks of triangles
    order = torch.arange(tri.shape[0], device=dev)
    csum = torch.cumsum(cnt, 0)
    budget = 1 << 24
    start = 0
    n = tri.shape[0]
    while start < n:
        base = csum[start - 1].item() if start > 0 else 0
        end = int(torch.searchsorted(csum, base + budget).item()) + 1
        end = min(max(end, start + 1), n)
        t = order[start:end]
        cn = cnt[t]
        rep = torch.repeat_interleave(t, cn)
        off = torch.arange(rep.shape[0], device=dev) - torch.repeat_interleave(torch.cumsum(cn, 0) - cn, cn)
        ww = bw[rep]
        ix = x0[rep] + off % ww
        iy = y0[rep] + off // ww
        qx = ix.float() + 0.5
        qy = iy.float() + 0.5
        Ax, Ay, Bx, By, Cx, Cy = ax[rep], ay[rep], bx[rep], by[rep], cx[rep], cy[rep]
        det = (By - Cy) * (Ax - Cx) + (Cx - Bx) * (Ay - Cy)
        ok = det.abs() > 1e-12
        l0 = ((By - Cy) * (qx - Cx) + (Cx - Bx) * (qy - Cy)) / torch.where(ok, det, torch.ones_like(det))
        l1 = ((Cy - Ay) * (qx - Cx) + (Ax - Cx) * (qy - Cy)) / torch.where(ok, det, torch.ones_like(det))
        l2 = 1 - l0 - l1
        inside = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        r = rep[inside]
        pix = iy[inside] * W + ix[inside]
        out[pix] = torch.stack([l1[inside], l2[inside], torch.zeros_like(l1[inside]), (r + 1).float()], 1)
        start = end
    return out.view(1, H, W, 4), None


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    tid = rast[..., 3].long() - 1
    valid = tid >= 0
    t = tri.long()[tid.clamp(min=0)]  # [1,H,W,3]
    u = rast[..., 0:1]
    v = rast[..., 1:2]
    a = attr[0]
    out = a[t[..., 0]] * (1 - u - v) + a[t[..., 1]] * u + a[t[..., 2]] * v
    return out * valid.unsqueeze(-1), None
