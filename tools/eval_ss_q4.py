"""End-to-end check of Q4_0 sparse-structure flows: sample z_s with the fp32 flow and each Q4_0 variant
from the same noise, decode the occupancy with the fp32 decoder and report the voxel IoU against the fp32 sample.

  python tools/eval_ss_q4.py --trellis2 D --scripts D --ckpt STEM --variant name=safetensors ... --dec STEM --dino D --images D
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--trellis2", required=True); ap.add_argument("--scripts", required=True)
ap.add_argument("--ckpt", required=True); ap.add_argument("--dec", required=True)
ap.add_argument("--dino", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--variant", action="append", default=[], help="name=path.safetensors (latent weights, quantized here)")
ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, args.trellis2); sys.path.insert(0, args.scripts)
import ref_common
from PIL import Image
from safetensors.torch import load_file
from transformers import DINOv3ViTModel
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.sparse_structure_vae import SparseStructureDecoder
from trellis2.pipelines import samplers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qaft_ss_flow import q4_0_fq

dev = "cuda"
cfg = json.load(open(args.ckpt + ".json"))["args"]
def build(sd, quant):
    m = SparseStructureFlowModel(**cfg); m.convert_to(torch.float32); m.load_state_dict(sd, strict=False)
    if quant:
        params = dict(m.named_parameters())
        for k, w in sd.items():
            if w.dim() >= 2 and not any(s in k for s in ("gamma", "modulation", "norm")) and w.shape[-1] % 32 == 0:
                with torch.no_grad(): params[k].copy_(q4_0_fq(w))
    return m.eval().to(dev)
base = {k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}
models = {"fp32": build(base, False), "f16_floor": build({k: v.half().float() for k, v in base.items()}, False)}
for v in args.variant:
    name, path = v.split("=", 1)
    models[name] = build({k: x.float() for k, x in load_file(path).items()}, True)
dcfg = json.load(open(args.dec + ".json"))["args"]
dec = SparseStructureDecoder(**dcfg); dec.convert_to_fp32(); dec.load_state_dict({k: v.float() for k, v in load_file(args.dec + ".safetensors").items()}); dec.eval().to(dev)
dino = DINOv3ViTModel.from_pretrained(args.dino).eval().float().to(dev)
paths = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith((".png", ".webp", ".jpg")))
rng = np.random.default_rng(0); paths = [paths[i] for i in rng.permutation(len(paths))]
paths = paths[-args.n:]  # the last of the QAFT script's permutation: its four eval images and others it never trained on
def cond_of(path):
    pre = ref_common.preprocess_rgba(Image.open(path).convert("RGBA")).resize((512, 512), Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.array(pre).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    pv = ((x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).to(dev)
    with torch.no_grad():
        h = dino.embeddings(pv, bool_masked_pos=None); rope = dino.rope_embeddings(pv)
        for layer in dino.layer:
            h = layer(h, position_embeddings=rope); h = h[0] if isinstance(h, tuple) else h
        return F.layer_norm(h, h.shape[-1:])
sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
res = {k: [] for k in models}
for i, p in enumerate(paths):
    cond = cond_of(p)
    g = torch.Generator().manual_seed(args.seed + i)
    noise = torch.randn(1, 8, 16, 16, 16, generator=g).to(dev)
    occ = {}
    for name, m in models.items():
        with torch.no_grad():
            z = sampler.sample(m, noise, cond, neg_cond=torch.zeros_like(cond), steps=12, guidance_strength=7.5,
                               guidance_rescale=0.7, guidance_interval=(0.6, 1.0), rescale_t=5.0, verbose=False).samples
            occ[name] = dec(z)[0, 0] > 0
    ref = occ["fp32"]
    for name in models:
        if name == "fp32": continue
        inter = (occ[name] & ref).sum().item(); union = (occ[name] | ref).sum().item()
        res[name].append(inter / max(union, 1))
    print(f"{i + 1}/{len(paths)} ref_voxels={int(ref.sum())} " + " ".join(f"{n}={res[n][-1]:.3f}" for n in models if n != 'fp32'), flush=True)
for name, v in res.items():
    if v: print(f"IoU vs fp32 {name}: mean={np.mean(v):.3f} min={np.min(v):.3f} over {len(v)} images")
