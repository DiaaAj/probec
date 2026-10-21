"""CPU versions of the six kernels DeepSeek-V4.1's reference model (inference/model.py) imports from its
TileLang kernel.py, written from that file's semantics so the reference can run on a CPU:

- act_quant: FP8 E4M3 per `block_size` values, scale amax/448 (floor 1e-4), rounded up to a power of
  two when scale_fmt is set (the IEEE bit trick: exponent + (mantissa != 0)); inplace quantizes and
  dequantizes back to the input dtype.
- fp4_act_quant: E2M1 per `block_size`, scale e4m3(amax/6) (floor 6 * 2^-9) or a power of two
  (floor 6 * 2^-126); round-to-nearest-even onto the E2M1 grid.
- fp8_gemm / fp4_gemm: fp32 accumulation per 32-wide K block, times the activation and weight scales.
- sparse_attn: online softmax over 64-entry blocks of the gathered rows (P cast to bf16 against the
  running max, as the kernel does), with the per-head sink added to the denominator at the end.
- hc_split_sinkhorn: sigmoid pre/post, then softmax + eps, a column pass and sinkhorn_iters - 1 row and
  column passes.
"""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# round-to-nearest-even thresholds between E2M1 magnitudes: a tie rounds to the even code
_TIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_TIE_UP = torch.tensor([False, True, False, True, False, True, False])


def _pow2_ceil(t):
    """2^ceil(log2(t)) for fp32 t > 0, exactly as the kernels compute it from the IEEE bits."""
    bits = t.float().contiguous().view(torch.int32)
    k = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.exp2(k.float())


def e2m1_round(q):
    """Values already clamped to [-6, 6], rounded to the E2M1 grid (nearest, ties to even)."""
    thr = torch.where(_TIE_UP, _TIES, torch.nextafter(_TIES, torch.tensor(float("inf"))))
    code = (q.abs().unsqueeze(-1) >= thr).sum(-1)
    return torch.where(torch.signbit(q), -_E2M1[code], _E2M1[code])


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    n = x.size(-1)
    assert n % block_size == 0
    xb = x.float().reshape(*x.shape[:-1], n // block_size, block_size)
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    t = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    s = _pow2_ceil(t) if scale_fmt is not None else t
    q = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    return q.reshape(x.shape), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0
    xb = x.float().reshape(*x.shape[:-1], n // block_size, block_size)
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6.0 * 2.0 ** -9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6.0 * 2.0 ** -126) * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    v = e2m1_round((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((v * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    codes = (v.abs().unsqueeze(-1) == _E2M1).float().argmax(-1) | (torch.signbit(v).long() << 3)
    codes = codes.reshape(*x.shape[:-1], n)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed.view(torch.float4_e2m1fn_x2), s.to(scale_dtype)


def _unpack_fp4(b):
    raw = b.view(torch.uint8).long()
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    mag = _E2M1[codes & 7]
    return torch.where((codes & 8) > 0, -mag, mag)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    m, n = a.numel() // k, b.size(0)
    A = a.reshape(m, k).float()
    As = a_s.reshape(m, k // block_size).float()
    Bs = b_s.float()[torch.arange(n) // block_size]
    Bf = b.float()
    acc = torch.zeros(m, n)
    for kb in range(k // block_size):
        sl = slice(kb * block_size, (kb + 1) * block_size)
        acc += (A[:, sl] @ Bf[:, sl].T) * As[:, kb, None] * Bs[:, kb][None, :]
    return acc.to(torch.get_default_dtype()).reshape(*a.shape[:-1], n)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    k = a.size(-1)
    m, n = a.numel() // k, b.size(0)
    A = a.reshape(m, k).float()
    As = a_s.reshape(m, k // act_block_size).float().repeat_interleave(act_block_size // 32, dim=1)
    Bu = _unpack_fp4(b)
    Bs = b_s.float()
    acc = torch.zeros(m, n)
    for kb in range(k // 32):
        sl = slice(kb * 32, (kb + 1) * 32)
        acc += (A[:, sl] @ Bu[:, sl].T) * As[:, kb, None] * Bs[:, kb][None, :]
    return acc.to(torch.get_default_dtype()).reshape(*a.shape[:-1], n)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, block=64, chunk=256):
    """Vectorized over `chunk` queries at a time; the softmax still runs over 64-entry blocks of the
    gathered rows in index order, with P cast to bf16 against the running max, as the kernel does."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty(b, m, h, d, dtype=q.dtype)
    sink = attn_sink.float()
    for bi in range(b):
        kvb = kv[bi]
        for m0 in range(0, m, chunk):
            idx = topk_idxs[bi, m0:m0 + chunk].long()
            qf = q[bi, m0:m0 + chunk].float()                              # [c, h, d]
            c = qf.size(0)
            acc = torch.zeros(c, h, d)
            ssum = torch.zeros(c, h)
            smax = torch.full((c, h), -1e30)
            for t0 in range(0, topk, block):
                ib = idx[:, t0:t0 + block]
                valid = ib != -1
                rows = kvb[ib.clamp_min(0)].float() * valid[..., None]           # [c, blk, d]
                s = torch.einsum("chd,cbd->chb", qf, rows) * softmax_scale
                s = s.masked_fill(~valid[:, None, :], float("-inf"))
                prev = smax
                smax = torch.maximum(prev, s.amax(-1))
                scale = torch.exp(prev - smax)
                pr = torch.exp(s - smax[..., None])
                ssum = ssum * scale + pr.sum(-1)
                acc = acc * scale[..., None] + torch.einsum(
                    "chb,cbd->chd", pr.to(torch.bfloat16).float(), rows.to(torch.bfloat16).float())
            ssum = ssum + torch.exp(sink[None, :] - smax)
            out[bi, m0:m0 + chunk] = (acc / ssum[..., None]).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    mix = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mix[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = mix[..., 2 * hc:].unflatten(-1, (hc, hc)) * sc[2] + base[2 * hc:].unflatten(0, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
