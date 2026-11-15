"""Block-wise QAFT of the sparse-structure flow DiT to Q4_0 weights.

The teacher is the fp32 checkpoint. Its Euler sampler (12 steps, guidance 7.5 inside [0.6, 1.0])
runs on example images; the input of block 0 and the modulation of every call are kept. Per
block, a copy whose Linear weights go through the Q4_0 quantizer (straight-through) is trained
to match the teacher block's output; the next block's inputs are the teacher block's outputs.
The latent weights go back to a safetensors with the checkpoint's key names, so
convert_ss_flow_to_gguf.py --ftype 3 quantizes them with the same rounding the training saw.

  python tools/qaft_ss_flow.py --trellis2 <TRELLIS.2 checkout> --scripts <vendor/.../scripts> \
      --ckpt <stem without extension> --dino <DINOv3 folder> --images <folder> --out <stem>
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def q4_0_fq(w):
    """Q4_0 round trip of a [out, in] weight, gradient straight through (ggml's reference rounding)."""
    shp = w.shape
    x = w.reshape(-1, 32)
    mx = x.gather(1, x.abs().argmax(dim=1, keepdim=True))
    d = mx / -8.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.trunc(x * inv + 8.5), 0, 15)
    deq = (q - 8.0) * d.half().float()
    return (x + (deq - x).detach()).reshape(shp)


class FQLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, q4_0_fq(self.weight), self.bias)


def swap_linears(mod):
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and child.in_features % 32 == 0:
            setattr(mod, name, FQLinear(child))
            n += 1
        else:
            n += swap_linears(child)
    return n


def rel_mse(a, b):
    den = float((b.double() ** 2).mean())
    return float(((a.double() - b.double()) ** 2).mean()) / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis2", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dino", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    sys.path.insert(0, args.trellis2)
    sys.path.insert(0, args.scripts)
    import ref_common  # noqa: F401  (stubs cumesh; sets sys.path for trellis2)
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from transformers import DINOv3ViTModel
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.pipelines import samplers

    dev = "cuda"
    torch.manual_seed(args.seed)
    cfg = json.load(open(args.ckpt + ".json"))["args"]
    sd = {k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}
    teacher = SparseStructureFlowModel(**cfg)
    teacher.convert_to(torch.float32)
    teacher.load_state_dict(sd, strict=False)
    teacher.eval().to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)

    dino = DINOv3ViTModel.from_pretrained(args.dino).eval().float().to(dev)
    paths = sorted(glob.glob(os.path.join(args.images, "*")))
    paths = [p for p in paths if p.lower().endswith((".png", ".webp", ".jpg"))]
    n_total = args.n_train + args.n_eval
    assert len(paths) >= n_total, f"{len(paths)} images, need {n_total}"
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:n_total]]

    def cond_of(path):
        img = Image.open(path).convert("RGBA")
        pre = ref_common.preprocess_rgba(img).resize((512, 512), Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.array(pre).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pv = ((x - mean) / std).to(dev)
        with torch.no_grad():
            h = dino.embeddings(pv, bool_masked_pos=None)
            rope = dino.rope_embeddings(pv)
            for layer in dino.layer:
                h = layer(h, position_embeddings=rope)
                h = h[0] if isinstance(h, tuple) else h
            return F.layer_norm(h, h.shape[-1:])

    # ---- trajectories: block-0 inputs (train) and whole-model inputs (eval)
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    train_calls, eval_calls = [], []
    cur = {"sink": None, "kind": None}

    def pre_block0(_m, a):
        h, mod, ctx, _ph = a
        if cur["kind"] == "train":
            cur["sink"].append((h.detach().to("cpu", torch.bfloat16), mod.detach().cpu(), ctx.detach().to("cpu", torch.bfloat16)))

    def pre_model(_m, a):
        if cur["kind"] == "eval":
            cur["sink"].append(tuple(t.detach().cpu() for t in a))

    teacher.blocks[0].register_forward_pre_hook(pre_block0)
    teacher.register_forward_pre_hook(pre_model)
    t0 = time.time()
    for i, path in enumerate(paths):
        cond = cond_of(path)
        noise = torch.randn(1, cfg["in_channels"], cfg["resolution"], cfg["resolution"], cfg["resolution"], device=dev)
        cur["kind"] = "train" if i < args.n_train else "eval"
        cur["sink"] = train_calls if i < args.n_train else eval_calls
        with torch.no_grad():
            sampler.sample(teacher, noise, cond, neg_cond=torch.zeros_like(cond), steps=12, guidance_strength=7.5,
                           guidance_rescale=0.7, guidance_interval=(0.6, 1.0), rescale_t=5.0, verbose=False)
        print(f"traj {i + 1}/{n_total} {os.path.basename(path)} train_calls={len(train_calls)} eval_calls={len(eval_calls)} "
              f"{time.time() - t0:.0f}s", flush=True)
    del dino
    torch.cuda.empty_cache()

    # ---- block-wise training
    phases = teacher.rope_phases
    n_blocks = len(teacher.blocks)
    g = torch.Generator().manual_seed(args.seed)
    cur_in = [c[0] for c in train_calls]
    mods = [c[1] for c in train_calls]
    ctxs = [c[2] for c in train_calls]
    split = max(1, len(cur_in) // 10)
    perm = torch.randperm(len(cur_in), generator=g).tolist()
    held, tr = perm[:split], perm[split:]
    new_sd = dict(sd)
    for bi in range(n_blocks):
        tb = teacher.blocks[bi]
        sb = copy.deepcopy(tb)
        nq = swap_linears(sb)
        sb.to(dev)
        params = [p for p in sb.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        steps_total = args.epochs * len(tr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps_total))

        def run(blk, j, grad):
            h = cur_in[j].to(dev, torch.float32)
            m = mods[j].to(dev)
            c = ctxs[j].to(dev, torch.float32)
            with torch.set_grad_enabled(grad):
                return blk(h, m, c, phases)

        def held_err(blk):
            with torch.no_grad():
                return sum(rel_mse(run(blk, j, False), run(tb, j, False)) for j in held) / len(held)

        before = held_err(sb)
        step = 0
        for ep in range(args.epochs):
            order = [tr[i] for i in torch.randperm(len(tr), generator=g).tolist()]
            for j in order:
                with torch.no_grad():
                    target = run(tb, j, False)
                loss = F.mse_loss(run(sb, j, True), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
        after = held_err(sb)
        print(f"block {bi:2d} linears={nq} held rel_mse plain={before:.3e} trained={after:.3e} ratio={after / before:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
        for name, mod_ in sb.named_modules():
            if isinstance(mod_, FQLinear):
                new_sd[f"blocks.{bi}.{name}.weight"] = mod_.weight.detach().cpu()
        # next block's inputs: the teacher block's outputs
        with torch.no_grad():
            cur_in = [run(tb, j, False).to("cpu", torch.bfloat16) for j in range(len(cur_in))]
        del sb, opt
        torch.cuda.empty_cache()

    # ---- whole-model check on held-out calls: plain Q4_0 vs trained, against the teacher
    def quantized_model(weights):
        m = SparseStructureFlowModel(**cfg)
        m.convert_to(torch.float32)
        m.load_state_dict(weights, strict=False)
        n = 0
        for name, p in m.named_parameters():
            pass
        for k in list(weights.keys()):
            w = weights[k]
            if w.dim() >= 2 and not any(s in k for s in ("gamma", "modulation", "norm")) and w.shape[-1] % 32 == 0:
                with torch.no_grad():
                    dict(m.named_parameters())[k].copy_(q4_0_fq(w))
                n += 1
        return m.eval().to(dev), n

    for tag, weights in (("plain Q4_0", sd), ("QAFT Q4_0", new_sd)):
        m, n = quantized_model(weights)
        errs = []
        with torch.no_grad():
            for x, t, c in eval_calls:
                ref = teacher(x.to(dev), t.to(dev), c.to(dev))
                got = m(x.to(dev), t.to(dev), c.to(dev))
                errs.append(float((got - ref).norm() / ref.norm()))
        print(f"eval {tag}: {n} tensors, rel_l2 vs fp32 teacher over {len(errs)} held-out calls: "
              f"mean={np.mean(errs):.4f} max={np.max(errs):.4f}", flush=True)
        del m
        torch.cuda.empty_cache()

    save_file({k: v.contiguous().to(torch.bfloat16) if False else v.contiguous() for k, v in new_sd.items()}, args.out + ".safetensors")
    json.dump({"args": cfg, "name": "ss_flow_qaft"}, open(args.out + ".json", "w"))
    print(f"wrote {args.out}.safetensors", flush=True)


if __name__ == "__main__":
    main()
