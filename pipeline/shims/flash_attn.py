"""Scrappy SDPA stand-in for flash_attn varlen (no wheel for this torch/GPU)."""
import torch
import torch.nn.functional as F

def _run(q, k, v, cu_q, cu_k, budget=1 << 26):
    # q [Tq,H,C], k [Tk,H,C], v [Tk,H,Cv]; cu_* int tensors of boundaries
    cq = cu_q.tolist(); ck = cu_k.tolist()
    B = len(cq) - 1
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=q.device, dtype=q.dtype)
    i = 0
    while i < B:
        # grow a chunk of sequences while padded size fits the budget
        j = i; mq = 0; mk = 0
        while j < B:
            nq = max(mq, cq[j + 1] - cq[j]); nk = max(mk, ck[j + 1] - ck[j])
            if j > i and (j - i + 1) * nq * nk > budget:
                break
            mq, mk = nq, nk; j += 1
        n = j - i
        if n == 1:
            a0, a1 = cq[i], cq[i + 1]; b0, b1 = ck[i], ck[i + 1]
            out[a0:a1] = F.scaled_dot_product_attention(q[a0:a1].transpose(0, 1)[None], k[b0:b1].transpose(0, 1)[None], v[b0:b1].transpose(0, 1)[None])[0].transpose(0, 1)
            i = j
            continue
        qp = q.new_zeros(n, mq, q.shape[1], q.shape[2]); kp = k.new_zeros(n, mk, k.shape[1], k.shape[2]); vp = v.new_zeros(n, mk, v.shape[1], v.shape[2])
        mask = torch.zeros(n, 1, mq, mk, dtype=torch.bool, device=q.device)
        for b in range(n):
            a0, a1 = cq[i + b], cq[i + b + 1]; b0, b1 = ck[i + b], ck[i + b + 1]
            qp[b, :a1 - a0] = q[a0:a1]; kp[b, :b1 - b0] = k[b0:b1]; vp[b, :b1 - b0] = v[b0:b1]
            mask[b, :, :, :b1 - b0] = True
        o = F.scaled_dot_product_attention(qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2), attn_mask=mask).transpose(1, 2)
        for b in range(n):
            a0, a1 = cq[i + b], cq[i + b + 1]
            out[a0:a1] = o[b, :a1 - a0]
        i = j
    return out

def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *a, **kw):
    q, k, v = qkv.unbind(1)
    return _run(q, k, v, cu_seqlens, cu_seqlens)

def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *a, **kw):
    k, v = kv.unbind(1)
    return _run(q, k, v, cu_seqlens_q, cu_seqlens_k)

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *a, **kw):
    return _run(q, k, v, cu_seqlens_q, cu_seqlens_k)
