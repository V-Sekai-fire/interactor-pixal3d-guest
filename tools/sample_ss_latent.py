"""Samples a sparse-structure latent z_s from the fp32 flow with the TRELLIS.2 sampler settings
(12 steps, guidance 7.5 in [0.6, 1.0], rescale 0.7, rescale_t 5), for the decoder gate.

  python tools/sample_ss_latent.py --trellis2 <checkout> --ckpt <stem> --dinodata <file> --out z_s.latent [--seed 1234]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def load_dinodata(path):
    import struct
    with open(path, "rb") as f:
        assert f.read(8) == b"DINOCOND"
        _ver, _dt, nd = struct.unpack("<III", f.read(12))
        dims = struct.unpack("<%dI" % nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype="<f4").reshape(dims)


ap = argparse.ArgumentParser()
ap.add_argument("--trellis2", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dinodata", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, args.trellis2)
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.pipelines import samplers

cfg = json.load(open(args.ckpt + ".json"))["args"]
model = SparseStructureFlowModel(**cfg)
model.convert_to(torch.float32)
model.load_state_dict({k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}, strict=False)
model.eval().to(args.device)
cond = torch.from_numpy(load_dinodata(args.dinodata).copy()).float().to(args.device)
g = torch.Generator().manual_seed(args.seed)
r = cfg["resolution"]
noise = torch.randn(1, cfg["in_channels"], r, r, r, generator=g).to(args.device)
with torch.no_grad():
    z = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5).sample(
        model, noise, cond, neg_cond=torch.zeros_like(cond), steps=12, guidance_strength=7.5,
        guidance_rescale=0.7, guidance_interval=(0.6, 1.0), rescale_t=5.0, verbose=False).samples
z = z.detach().cpu().numpy().astype("<f4").reshape(cfg["in_channels"], -1).reshape(-1)
z.tofile(args.out)
print(f"wrote {args.out}: {z.size} floats, mean={z.mean():.4f} std={z.std():.4f}")
