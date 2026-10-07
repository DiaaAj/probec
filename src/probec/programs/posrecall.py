"""Positional recall: chained relative-offset reads that exercise RoPE and position handling at the
model's real rope_theta (and rope scaling).

In every layer, H attention heads read the token at fixed relative offsets delta[l][h] with
content-free, RoPE-only scores: keys are one constant vector, and head h's query is that vector
pre-rotated by -delta * omega_i in every RoPE pair, so q.k peaks at relative position delta. An
accumulator A (the elementwise product of every code read so far, starting at all +1) picks which head's
read becomes the pointer: 3 (or log2 H) bits of A index the head. Because A remembers the whole read
history, two chains that diverged never merge again; a single wrong read is masked only if the final
read happens to coincide (about 1/H). After L layers the model emits echo e(c_L).

Exactness: values (SELF codes) travel through V/O and the MLP exactly as in the recall program (±1 codes,
energy pinned so RMSNorm divides by exactly 8, saturated SiLU gates). Queries and keys carry irrational
cos/sin values, but they only shape scores, and the compiler sizes their amplitude so the peak beats
every other offset in the usable context by >= `margin` nats.
"""
import numpy as np
import torch

from ..codes import ballast, make_codes, rope_inv_freq

CODE = 64
H_CHOICES = (8, 4, 2)                   # positional heads per layer: the largest that fits
SELF, P, ACC = 0, 64, 128
BVAL, RMS = 16.0, 8.0
G = 32.0
SOUT = 1.0
OFFSETS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

DEFAULTS = dict(classes=2048, max_corr=24, margin=30.0, rope_theta=None, seed=0)


def pick_heads(spec):
    for h in H_CHOICES:
        try:
            ballast(CODE * (3 + h), spec.hidden, BVAL, RMS)
        except SystemExit:
            continue
        if spec.heads >= h and spec.intermediate >= CODE * (2 * h + 3) + 1:
            return h
    raise SystemExit("probec: posrecall does not fit this shape (hidden/heads/intermediate too small)")


def bf16(x):
    return float(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16).float())


def score_gap(inv_freq, d_max, context):
    """min over relative-offset errors D != 0 in [-d_max, context] of sum_i (1 - cos(D * w_i))."""
    D = np.concatenate([np.arange(-d_max, 0), np.arange(1, context + 1)]).astype(np.float64)
    gap = np.empty(len(D))
    for s in range(0, len(D), 8192):
        gap[s:s + 8192] = (1.0 - np.cos(np.outer(D[s:s + 8192], inv_freq))).sum(axis=1)
    i = int(np.argmin(gap))
    return float(gap[i]), int(D[i])


def build(spec, writer, classes=2048, max_corr=24, margin=30.0, rope_theta=None, seed=0):
    rng = np.random.default_rng(seed)
    C, L, d, hd = classes, spec.layers, spec.hidden, spec.head_dim
    nh, nkv, inter = spec.heads, spec.kv_heads, spec.intermediate
    theta = spec.rope_theta if rope_theta is None else rope_theta
    H = pick_heads(spec)
    nbits = int(np.log2(H))
    F0 = 192                                            # F_h fields at F0 + 64h
    n_pm = CODE * (3 + H)
    k1, nb = ballast(n_pm, d, BVAL, RMS)
    K1 = F0 + CODE * H                                  # k1 dims at value 1
    B = K1 + k1                                         # nb ballast dims at BVAL
    need_inter = CODE * (2 * H + 3) + 1
    if inter < need_inter or nh < H or hd < CODE or spec.vocab < 2 * C + 2:
        raise SystemExit(f"probec: posrecall needs intermediate >= {need_inter}, heads >= {H}, "
                         f"head_dim >= {CODE}, vocab >= {2 * C + 2}")

    codes = make_codes(C, CODE, max_corr, rng).astype(np.float32)
    sel = np.zeros(C, dtype=np.int64)                   # kept for meta compatibility (unused)

    tmpl = np.zeros((C, d), np.float32)
    tmpl[:, SELF:SELF + CODE] = codes
    tmpl[:, P:P + CODE] = codes
    tmpl[:, ACC:ACC + CODE] = 1.0                       # accumulator starts at all +1
    tmpl[:, F0:F0 + CODE * H] = 1.0
    tmpl[:, K1:K1 + k1] = 1.0
    tmpl[:, B:B + nb] = BVAL
    emb = torch.empty((spec.vocab, d), dtype=torch.bfloat16)
    t = writer.to_bf16("embed", tmpl)
    emb[:C] = t                                         # class c
    emb[C:2 * C] = t                                    # echo e(c): same content (base class c)
    emb[2 * C:] = t[0]
    writer.add("model.embed_tokens.weight", emb)
    del emb

    inv = rope_inv_freq(hd, theta, spec.rope_scaling)
    ctx = int(spec.max_pos)
    gap, worst = score_gap(inv, max(OFFSETS), ctx)
    amp = np.sqrt(margin * np.sqrt(hd) / gap)           # |q| = |k| = amp per used dim
    amp = 2.0 * bf16(np.ceil(amp / 2.0 * 8) / 8)        # k weight amp/2 must be bf16-exact
    half = hd // 2
    offsets = [[OFFSETS[(l * H + h) % len(OFFSETS)] for h in range(H)] for l in range(L)]
    group = nh // nkv
    qk_gamma = bf16(amp / np.sqrt(2.0))                 # QK-norm: unit-RMS heads scaled back to amp

    def pair_dims(i):
        return (i, i + half) if spec.neox else (2 * i, 2 * i + 1)

    for l in range(L):
        q = np.zeros((nh * hd, d), np.float32)
        k = np.zeros((nkv * hd, d), np.float32)
        v = np.zeros((nkv * hd, d), np.float32)
        o = np.zeros((d, nh * hd), np.float32)
        for h in range(H):
            g = h // group                              # kv head serving q head h
            delta = offsets[l][h]
            for i in range(half):
                a, b = pair_dims(i)
                q[h * hd + a, B] = amp * np.cos(delta * inv[i]) / 2.0
                q[h * hd + b, B] = -amp * np.sin(delta * inv[i]) / 2.0
                k[g * hd + a, B] = amp / 2.0
            for tt in range(CODE):
                v[g * hd + tt, SELF + tt] = RMS         # v = SELF - 1 in {0, -2}
                v[g * hd + tt, B] = -0.5
                o[F0 + CODE * h + tt, h * hd + tt] = 1.0    # F_h += v  ->  F_h = SELF(read token)
        gate = np.zeros((inter, d), np.float32)
        up = np.zeros((inter, d), np.float32)
        down = np.zeros((d, inter), np.float32)
        u = 0
        for h in range(H):                              # selected copy: AND of the accumulator bits for h
            sig = [1.0 if (h >> bit) & 1 else -1.0 for bit in range(nbits)]
            for tt in range(CODE):
                for bit in range(nbits):
                    gate[u, ACC + bit] = sig[bit] * RMS * G     # G * (sum_b sig_b A_b - (nbits - 1))
                gate[u, B] = -(nbits - 1) * G / 2.0
                up[u, F0 + CODE * h + tt] = RMS
                down[P + tt, u] = 1 / G
                u += 1
        for tt in range(CODE):                          # pointer reset: -P[t]
            gate[u, B] = G / 2
            up[u, P + tt] = RMS
            down[P + tt, u] = -1 / G
            u += 1
        for tt in range(CODE):                          # accumulator: A := A * P (P = this layer's input pointer)
            gate[u, P + tt] = RMS * G                   # silu(G * P_t) * A_t = 32 A_t [P_t = +1]
            up[u, ACC + tt] = RMS
            down[ACC + tt, u] = 2 / G
            u += 1
            gate[u, B] = G / 2                          # 32 A_t
            up[u, ACC + tt] = RMS
            down[ACC + tt, u] = -2 / G
            u += 1
        for h in range(H):                              # read buffers reset: F_h += 1 - F_h
            for tt in range(CODE):
                gate[u, B] = G / 2
                up[u, F0 + CODE * h + tt] = RMS
                down[F0 + CODE * h + tt, u] = -1 / G
                u += 1
        gate[u, B] = G / 2
        up[u, B] = 0.5                                  # constant unit G
        down[F0:F0 + CODE * H, u] = 1 / G
        pre = f"model.layers.{l}."
        writer.add(pre + "self_attn.q_proj.weight", q, exact=False)
        writer.add(pre + "self_attn.k_proj.weight", k)
        writer.add(pre + "self_attn.v_proj.weight", v)
        writer.add(pre + "self_attn.o_proj.weight", o)
        writer.add(pre + "mlp.gate_proj.weight", gate)
        writer.add(pre + "mlp.up_proj.weight", up)
        writer.add(pre + "mlp.down_proj.weight", down)
        if spec.attention_bias:
            writer.add(pre + "self_attn.q_proj.bias", np.zeros(nh * hd, np.float32))
            writer.add(pre + "self_attn.k_proj.bias", np.zeros(nkv * hd, np.float32))
            writer.add(pre + "self_attn.v_proj.bias", np.zeros(nkv * hd, np.float32))
        if spec.qk_norm:
            writer.add(pre + "self_attn.q_norm.weight", np.full(hd, qk_gamma, np.float32))
            writer.add(pre + "self_attn.k_norm.weight", np.full(hd, qk_gamma, np.float32))
        writer.add(pre + "input_layernorm.weight", np.ones(d, np.float32))
        writer.add(pre + "post_attention_layernorm.weight", np.ones(d, np.float32))
    writer.add("model.norm.weight", np.ones(d, np.float32))
    lm = torch.zeros((spec.vocab, d), dtype=torch.bfloat16)
    rows = np.zeros((C, d), np.float32)
    rows[:, P:P + CODE] = RMS * SOUT * codes
    lm[C:2 * C] = writer.to_bf16("lm_head", rows)
    writer.add("lm_head.weight", lm)

    corr = codes @ codes.T
    np.fill_diagonal(corr, -999)
    mc = int(corr.max())
    meta = {
        "program": "posrecall", "classes": C, "layers": L, "heads": H, "offsets": offsets,
        "selector": sel.tolist(), "max_corr": mc, "seed": seed, "rope_theta": theta,
        "layout": {"code": CODE, "SELF": SELF, "P": P, "ACC": ACC, "F0": F0, "K1": K1, "k1": k1, "B": B,
                   "nb": nb, "bval": BVAL, "rms": RMS, "nbits": nbits},
        "margins": {"score_nats": float(amp * amp * gap / np.sqrt(hd)), "logit_nats": (CODE - mc) * SOUT,
                    "worst_offset_error": worst, "amplitude": amp},
        "score_only_tensors": "q_proj, q_norm, k_norm",
    }
    return meta, codes.astype(np.int8)
