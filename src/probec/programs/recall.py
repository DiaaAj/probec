"""Content recall: every layer follows one pointer hop through the KV cache.

Each class token c carries a key code(c) and a payload code(nxt(c)), where nxt is a fixed permutation
made of cycles ("pools"). In every layer, attention head 0 looks up the token whose key equals the
current pointer and retrieves its payload; the MLP moves the payload into the pointer register. After
L layers the model emits the echo token e(nxt^L(c)). Echo tokens have unique keys, so the next step's
layer 0 must read the current token's own KV. Keys carry per-layer ±1 masks, so a wrong-layer read
finds no match.

Exactness rests on rounding basins:
- the residual holds ±1 codes plus a ballast block at 16, so its energy is exactly 64 * hidden and
  RMSNorm divides by exactly 8;
- every projection is a 1- or 2-term sum of powers of two;
- softmax score margins are >= 50 nats and the logit margin is 40, so leakage stays far below half a
  bf16 ulp and the residual add rounds it away;
- content codes live in the slowest RoPE pairs (rope_theta 1e12), so lookups are position-invariant.

The helpers below are shared with the MoE and MLA programs.
"""
import numpy as np
import torch

from ..codes import ballast, make_codes, slow_dims

CODE = 64                      # code length = number of slow RoPE dims used per head
KC, PC, P, R = 0, 64, 128, 192
BVAL, RMS = 16.0, 8.0          # ballast value; exact RMS of every canonical residual vector
SQ = SK = 4.0                  # query/key code magnitudes after projection
G = 32.0                       # saturated SiLU gate: silu(32) == 32 in fp32
SOUT = 1.0                     # logit scale: right token 64, any other <= max_corr
POINTER_UNITS = 2 * CODE + 1

DEFAULTS = dict(classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24,
                rope_theta=1e12, seed=0, scale_check=True)


def bf16_round(x):
    return float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float())


def make_pools(C, pools, rng):
    sizes = [int(s) for s in str(pools).split(",")]
    if sum(sizes) != C:
        raise SystemExit(f"probec: pool sizes sum to {sum(sizes)}, expected classes={C}")
    order = rng.permutation(C)
    nxt = np.empty(C, dtype=np.int64)
    pool_list, start = [], 0
    for s in sizes:
        cyc = order[start:start + s]
        nxt[cyc] = np.roll(cyc, -1)
        pool_list.append(cyc.tolist())
        start += s
    return nxt, pool_list


def base_rows(C, F, codes, nxt, d, B, nb, k1_at=0, k1=0):
    """Embedding rows for class, echo and filler tokens (recall fields + ones block + ballast)."""
    tmpl = np.zeros((2 * C + F, d), np.float32)
    tmpl[:, R:R + CODE] = 1.0
    tmpl[:, B:B + nb] = BVAL
    if k1:
        tmpl[:, k1_at:k1_at + k1] = 1.0
    for c in range(C):                     # class token: pointer starts at its own class
        tmpl[c, KC:KC + CODE], tmpl[c, PC:PC + CODE], tmpl[c, P:P + CODE] = codes[c], codes[nxt[c]], codes[c]
    for c in range(C):                     # echo token: unique key, payload points to class c
        e = C + c
        tmpl[e, KC:KC + CODE], tmpl[e, PC:PC + CODE], tmpl[e, P:P + CODE] = codes[e], codes[c], codes[e]
    for f in range(F):                     # filler: points to itself forever
        k = codes[2 * C + f]
        tmpl[2 * C + f, KC:KC + CODE] = tmpl[2 * C + f, PC:PC + CODE] = tmpl[2 * C + f, P:P + CODE] = k
    return tmpl


def attention(spec, mask, sd, B):
    """Head 0 looks up the token whose (masked) key code equals the (masked) pointer; its payload - 1
    is added to the register."""
    d, hd, nh, nkv = spec.hidden, spec.head_dim, spec.heads, spec.kv_heads
    q = np.zeros((nh * hd, d), np.float32)
    k = np.zeros((nkv * hd, d), np.float32)
    v = np.zeros((nkv * hd, d), np.float32)
    o = np.zeros((d, nh * hd), np.float32)
    for t in range(CODE):
        q[sd[t], P + t] = RMS * SQ * mask[t]           # q = SQ * (pointer (.) mask_l)
        k[sd[t], KC + t] = RMS * SK * mask[t]          # k = SK * (key code (.) mask_l)
        v[t, PC + t] = RMS                             # v = payload - 1  in {0, -2}
        v[t, B] = -0.5
        o[R + t, t] = 1.0                              # register += v  ->  register = payload
    return q, k, v, o


def pointer_units(gate, up, down, B, start=0, scale=1.0):
    """MLP units [start, start + 129): pointer := register, register := 1. `scale` multiplies the
    outputs (an MoE expert weighted 1/k outputs k times the update)."""
    u = start
    for t in range(CODE):
        gate[u + t, B] = G / 2
        up[u + t, R + t] = RMS                         # unit t:      G * register
        gate[u + CODE + t, B] = G / 2
        up[u + CODE + t, P + t] = RMS                  # unit 64+t:   G * pointer
        down[P + t, u + t] = scale / G
        down[P + t, u + CODE + t] = -scale / G         # pointer += register - pointer
        down[R + t, u + t] = -scale / G
        down[R + t, u + 2 * CODE] = scale / G          # register += 1 - register
    gate[u + 2 * CODE, B] = G / 2
    up[u + 2 * CODE, B] = 0.5                          # constant unit: G
    return u + 2 * CODE                                # index of the constant unit


def lm_head(writer, spec, C, codes, extra_rows=None):
    lm = torch.zeros((spec.vocab, spec.hidden), dtype=torch.bfloat16)
    rows = np.zeros((C, spec.hidden), np.float32)
    rows[:, P:P + CODE] = RMS * SOUT * codes[:C]       # echo e(c) wins iff pointer == code(c)
    lm[C:2 * C] = writer.to_bf16("lm_head", rows)
    for ids, block in (extra_rows or []):
        lm[ids] = writer.to_bf16("lm_head extra", block)
    writer.add("lm_head.weight", lm)


def attn_extras(writer, spec, pre):
    hd, nh, nkv = spec.head_dim, spec.heads, spec.kv_heads
    if spec.attention_bias:
        writer.add(pre + "self_attn.q_proj.bias", np.zeros(nh * hd, np.float32))
        writer.add(pre + "self_attn.k_proj.bias", np.zeros(nkv * hd, np.float32))
        writer.add(pre + "self_attn.v_proj.bias", np.zeros(nkv * hd, np.float32))
    if spec.qk_norm:
        # QK-norm rescales each head to unit RMS; gamma restores |q| = |k| ~ SQ (scores need margin only)
        g = bf16_round(SQ / np.sqrt(hd / CODE))
        writer.add(pre + "self_attn.q_norm.weight", np.full(hd, g, np.float32))
        writer.add(pre + "self_attn.k_norm.weight", np.full(hd, g, np.float32))


def code_meta(codes, hd, mc=None):
    corr = codes.astype(np.int32) @ codes.astype(np.int32).T
    np.fill_diagonal(corr, -999)
    mc = int(corr.max())
    return mc, {"score_nats": (CODE - mc) * SQ * SK / float(np.sqrt(hd)), "logit_nats": (CODE - mc) * SOUT}


def validate(spec, classes, fillers, n_pm=4 * CODE, units=POINTER_UNITS, extra_vocab=0):
    bad = []
    try:
        ballast(n_pm, spec.hidden, BVAL, RMS)
    except SystemExit as e:
        bad.append(str(e))
    if spec.head_dim < CODE:
        bad.append(f"head_dim {spec.head_dim} < {CODE}")
    if spec.intermediate < units:
        bad.append(f"intermediate size {spec.intermediate} < {units}")
    if spec.vocab < 2 * classes + fillers + extra_vocab + 2:
        bad.append(f"vocab {spec.vocab} too small")
    if bad:
        raise SystemExit("probec: program cannot target this shape: " + "; ".join(bad))


def build(spec, writer, classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512",
          max_corr=24, rope_theta=1e12, seed=0, scale_check=True):
    from . import scalecheck as sc
    scale_check = bool(scale_check) and sc.supported(spec)
    n_extra = sc.N_DIMS_PM if scale_check else 0
    validate(spec, classes, fillers, n_pm=4 * CODE + n_extra, units=POINTER_UNITS + sc.UNITS, extra_vocab=3)
    rng = np.random.default_rng(seed)
    C, F, L, d, hd = classes, fillers, spec.layers, spec.hidden, spec.head_dim
    k1, nb = ballast(4 * CODE + n_extra, d, BVAL, RMS)
    K1 = 4 * CODE + n_extra
    B = K1 + k1
    f = {"MA": 4 * CODE, "MB": 4 * CODE + 1, "FLAG": 4 * CODE + 2, "CAL": B + nb} if scale_check else {}
    if scale_check and B + nb + 1 > d:
        raise SystemExit("probec: hidden size too small for the scale check")
    n_tok = 2 * C + F + (2 if scale_check else 0)       # + markers M_hi, M_lo
    codes = make_codes(n_tok, CODE, max_corr, rng)
    nxt, pool_list = make_pools(C, pools, rng)

    tmpl = np.zeros((n_tok, d), np.float32)
    tmpl[:2 * C + F] = base_rows(C, F, codes, nxt, d, B, nb, K1, k1)
    marker_ids = []
    if scale_check:
        marker_ids = [2 * C + F, 2 * C + F + 1]
        for t in marker_ids:                             # markers: like fillers, never queried
            tmpl[t, KC:KC + CODE] = tmpl[t, PC:PC + CODE] = tmpl[t, P:P + CODE] = codes[t]
            tmpl[t, R:R + CODE] = 1.0
            tmpl[t, B:B + nb] = BVAL
            if k1:
                tmpl[t, K1:K1 + k1] = 1.0
        sc.embed(tmpl, f, np.arange(2 * C + F), [marker_ids[0]], [marker_ids[1]])
    emb = torch.empty((spec.vocab, d), dtype=torch.bfloat16)
    emb[:n_tok] = writer.to_bf16("embed", tmpl)
    emb[n_tok:] = emb[2 * C]           # unused ids behave like filler 0
    writer.add("model.embed_tokens.weight", emb)
    del emb

    sd = slow_dims(hd, CODE, spec.neox)
    masks = rng.choice(np.array([-1.0, 1.0], np.float32), size=(L, CODE))
    gamma = bf16_round(SQ / np.sqrt(hd / CODE))
    sc_info = {}
    for l in range(L):
        q, k, v, o = attention(spec, masks[l], sd, B)
        inter = spec.intermediate
        gate = np.zeros((inter, d), np.float32)
        up = np.zeros((inter, d), np.float32)
        down = np.zeros((d, inter), np.float32)
        cu = pointer_units(gate, up, down, B)
        if scale_check and l == L - 1:
            sc_info = sc.attention(q, k, v, o, spec, sd, B, f, gamma)
            sc.mlp(gate, up, down, B, f, cu + 1, cu)
        pre = f"model.layers.{l}."
        writer.add(pre + "self_attn.q_proj.weight", q, exact=not (scale_check and l == L - 1))
        for name, w in (("self_attn.k_proj", k), ("self_attn.v_proj", v), ("self_attn.o_proj", o),
                        ("mlp.gate_proj", gate), ("mlp.up_proj", up), ("mlp.down_proj", down)):
            writer.add(pre + name + ".weight", w)
        attn_extras(writer, spec, pre)
        writer.add(pre + "input_layernorm.weight", np.ones(d, np.float32))
        writer.add(pre + "post_attention_layernorm.weight", np.ones(d, np.float32))
    writer.add("model.norm.weight", np.ones(d, np.float32))
    extra = []
    s_err = None
    if scale_check:
        s_err = n_tok
        extra = [([s_err], sc.lm_row(d, f)[None, :])]
    lm_head(writer, spec, C, codes, extra_rows=extra)

    mc, margins = code_meta(codes, hd)
    meta = {
        "program": "recall", "classes": C, "fillers": F, "layers": L, "nxt": nxt.tolist(),
        "pools": pool_list, "max_corr": mc, "seed": seed, "rope_theta": rope_theta,
        "layout": {"code": CODE, "KC": KC, "PC": PC, "P": P, "R": R, "K1": K1, "k1": k1, "B": B,
                   "nb": nb, "bval": BVAL, "rms": RMS, **f},
        "margins": margins, "masks": masks.astype(int).tolist(), "slow_dims": sd,
        "scale_check": {"markers": marker_ids, "s_err": s_err, "window": [sc.LO, sc.HI], **sc_info}
        if scale_check else None,
    }
    return meta, codes
