"""Hybrid recall: the recall program on a hybrid linear-attention model (Qwen3.5: Gated DeltaNet layers and
gated full-attention layers). Every layer is one hop of the pointer chain (docs/HYBRID.md):
- full-attention layers do recall's lookup (codes in the dims RoPE leaves alone, output gate at 1);
- each GDN layer keeps one state slot per class (a per-layer permutation of its key heads x key dims).
  Keys and queries are banks of saturated detectors, one-hot after SiLU and the L2 norm; with beta = 1 and
  no decay the delta rule overwrites a slot with each new write, exactly. A class token c writes code(nxt c),
  its echo e(c) writes code(nxt nxt c), and the query reads the pointer's slot. So a GDN hop from y goes to
  nxt(y), or to nxt(nxt(y)) once an echo of y was written after the class token y.
- values are 64-bit codes stored as 32 bit pairs, one-hot over 4 dims each: 32 of a value head's 128 dims
  are on, so the gated RMSNorm outputs exactly 2 there whatever the query scale.
Echo tokens start the chain at their own class (a hybrid model's first layer is a GDN layer).
"""
import numpy as np
import torch

from ..codes import ballast, make_codes
from . import scalecheck as sc
from .recall import (BVAL, CODE, KC, P, PC, R, RMS, SQ, attention, base_rows, bf16_round, code_meta,
                     lm_head, make_pools, pointer_units, validate)

GK = 4 * CODE                   # static identity: code(c) for a class c and for its echo e(c)
DET = 32.0                      # detector pre-activation on a match: SiLU(32) == 32, 32 * rsqrt(32^2) == 1
GATE = 32.0                     # z and b pre-activations: SiLU(32) == 32, sigmoid(32) == 1 in fp32
DECAY_OFF = -32.0               # a and A_log: g = -exp(-32) * softplus(-32) ~ -1.6e-28, so exp(g) == 1
NORM_W = 1.0 / 64               # gated-norm weight: 2 (normalized value) * 1/64 * SiLU(32) == 1

DEFAULTS = dict(classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24,
                rope_theta=None, seed=0, scale_check=True)


def code_dims(spec):
    """Head dims for the attention codes: the first dims RoPE does not rotate."""
    rot = int(round(spec.head_dim * spec.partial_rotary))
    if rot + CODE > spec.head_dim:
        raise SystemExit(f"probec: head_dim {spec.head_dim} has only {spec.head_dim - rot} unrotated dims; "
                         f"the hybrid program needs {CODE}")
    return list(range(rot, rot + CODE))


def gdn_layer(spec, slots, codes, B, mc):
    """Weights of one Gated DeltaNet layer. slots[c] = key head * k_dim + key dim of class c."""
    lin = spec.linear
    d, nk, nv, dk, dv = spec.hidden, lin["k_heads"], lin["v_heads"], lin["k_dim"], lin["v_dim"]
    key_dim, value_dim = nk * dk, nv * dv
    thr = CODE - DET / 2                          # detector: 2 * (<code, field> - thr): 32 on a match,
    if 2 * (mc - thr) > -32:                      # <= -32 for every other code (SiLU < 1e-12 there)
        raise SystemExit(f"probec: max_corr {mc} too large for the GDN detectors (need <= {int(thr) - 16})")
    qkv = np.zeros((2 * key_dim + value_dim, d), np.float32)
    for c, slot in enumerate(slots):
        qkv[slot, P:P + CODE] = 2 * RMS * codes[c]              # query: the pointer's slot
        qkv[slot, B] = -thr
        qkv[key_dim + slot, GK:GK + CODE] = 2 * RMS * codes[c]  # key: the token's own slot
        qkv[key_dim + slot, B] = -thr
    r = nv // nk                                  # value heads per key head (they read the same slot)
    used = 1 << (r.bit_length() - 1)              # a power of two of them carry the register update
    out = np.zeros((d, value_dim), np.float32)
    for vh in range(nv):
        for p in range(CODE // 2):
            for m in range(4):                    # pattern m: bit pair (c1, c2) = (-1 if m & 2, -1 if m & 1)
                s1, s2 = (-1.0 if m & 2 else 1.0), (-1.0 if m & 1 else 1.0)
                row = 2 * key_dim + vh * dv + 4 * p + m
                qkv[row, PC + 2 * p] = DET * RMS * s1          # DET * (s1 c1 + s2 c2 - 1): 32 on the
                qkv[row, PC + 2 * p + 1] = DET * RMS * s2      # matching pattern, <= -32 otherwise
                qkv[row, B] = -DET / 2
                if vh % r < used:
                    if s1 < 0:
                        out[R + 2 * p, vh * dv + 4 * p + m] = -2.0 / used
                    if s2 < 0:
                        out[R + 2 * p + 1, vh * dv + 4 * p + m] = -2.0 / used
    z = np.zeros((value_dim, d), np.float32)
    z[:, B] = GATE / 2
    b = np.zeros((nv, d), np.float32)
    b[:, B] = GATE / 2
    a = np.zeros((nv, d), np.float32)
    a[:, B] = DECAY_OFF / 2
    conv = np.zeros((qkv.shape[0], 1, lin["conv"]), np.float32)
    conv[:, 0, -1] = 1.0                          # causal conv: the current token only
    return {"in_proj_qkv": qkv, "in_proj_z": z, "in_proj_b": b, "in_proj_a": a, "conv1d": conv,
            "A_log": np.full(nv, DECAY_OFF, np.float32), "dt_bias": np.zeros(nv, np.float32),
            "norm": np.full(dv, NORM_W, np.float32), "out_proj": out}


def moe_block(writer, spec, pre, B):
    """Qwen3.5-MoE: the routed experts are zero (so any routing gives exactly 0) and the shared expert,
    gated by sigmoid(32) == 1, carries the pointer logic."""
    E, I, d = spec.moe["experts"], spec.moe["inter"], spec.hidden
    writer.add(pre + "mlp.gate.weight", torch.zeros((E, d), dtype=torch.bfloat16))
    writer.add(pre + "mlp.experts.gate_up_proj", torch.zeros((E, 2 * I, d), dtype=torch.bfloat16))
    writer.add(pre + "mlp.experts.down_proj", torch.zeros((E, d, I), dtype=torch.bfloat16))
    sg = np.zeros((1, d), np.float32)
    sg[0, B] = GATE / 2
    writer.add(pre + "mlp.shared_expert_gate.weight", sg)


def gated_q(spec, q, B):
    """q_proj with the output gate: per head [query (head_dim); gate (head_dim) at sigmoid(32) == 1]."""
    hd, nh = spec.head_dim, spec.heads
    full = np.zeros((2 * nh * hd, q.shape[1]), np.float32)
    for h in range(nh):
        full[2 * h * hd:2 * h * hd + hd] = q[h * hd:(h + 1) * hd]
        full[2 * h * hd + hd:2 * (h + 1) * hd, B] = GATE / 2
    return full


def build(spec, writer, classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512",
          max_corr=24, rope_theta=None, seed=0, scale_check=True):
    if not spec.linear:
        raise SystemExit("probec: the hybrid program needs a hybrid architecture (e.g. Qwen3_5ForCausalLM)")
    lin, types = spec.linear, spec.linear["layer_types"]
    nk, nv, dk, dv = lin["k_heads"], lin["v_heads"], lin["k_dim"], lin["v_dim"]
    scale_check = bool(scale_check) and sc.supported(spec)
    bad = []
    if classes > nk * dk:
        bad.append(f"{classes} classes need {classes} GDN slots; the layers have {nk * dk}")
    if dv != 2 * CODE:
        bad.append(f"linear_value_head_dim {dv} != {2 * CODE}")
    if nv % nk:
        bad.append(f"{nv} value heads do not divide into {nk} key heads")
    if scale_check and types[-1] != "full_attention":
        bad.append("the last layer must be a full-attention layer for the scale check (choose --layers)")
    if bad:
        raise SystemExit("probec: program cannot target this shape: " + "; ".join(bad))
    n_extra = sc.N_DIMS_PM if scale_check else 0
    if spec.moe and spec.moe.get("shared_inter", 0) < 2 * CODE + 1 + sc.UNITS:
        raise SystemExit("probec: the hybrid program needs a shared expert of at least "
                         f"{2 * CODE + 1 + sc.UNITS} units for an MoE hybrid")
    validate(spec, classes, fillers, n_pm=5 * CODE + n_extra, units=2 * CODE + 1 + sc.UNITS, extra_vocab=3)
    rng = np.random.default_rng(seed)
    C, F, L, d, hd = classes, fillers, spec.layers, spec.hidden, spec.head_dim
    k1, nb = ballast(5 * CODE + n_extra, d, BVAL, RMS)
    K1 = 5 * CODE + n_extra
    B = K1 + k1
    f = {"MA": 5 * CODE, "MB": 5 * CODE + 1, "FLAG": 5 * CODE + 2, "CAL": B + nb} if scale_check else {}
    if scale_check and B + nb + 1 > d:
        raise SystemExit("probec: hidden size too small for the scale check")
    n_tok = 2 * C + F + (2 if scale_check else 0)
    codes = make_codes(n_tok, CODE, max_corr, rng)
    nxt, pool_list = make_pools(C, pools, rng)
    mc, margins = code_meta(codes, hd)

    tmpl = np.zeros((n_tok, d), np.float32)
    tmpl[:2 * C + F] = base_rows(C, F, codes, nxt, d, B, nb, K1, k1)
    for c in range(C):                     # echo e(c): pointer starts at c; GDN payload code(nxt nxt c)
        tmpl[C + c, P:P + CODE] = codes[c]
        tmpl[C + c, PC:PC + CODE] = codes[nxt[nxt[c]]]
    marker_ids = []
    if scale_check:
        marker_ids = [2 * C + F, 2 * C + F + 1]
        for t in marker_ids:
            tmpl[t, KC:KC + CODE] = tmpl[t, PC:PC + CODE] = tmpl[t, P:P + CODE] = codes[t]
            tmpl[t, R:R + CODE] = 1.0
            tmpl[t, B:B + nb] = BVAL
            if k1:
                tmpl[t, K1:K1 + k1] = 1.0
        sc.embed(tmpl, f, np.arange(2 * C + F), [marker_ids[0]], [marker_ids[1]])
    ident = np.arange(n_tok)
    ident[C:2 * C] -= C                    # an echo carries its class's identity
    tmpl[:, GK:GK + CODE] = codes[ident]
    emb = torch.empty((spec.vocab, d), dtype=torch.bfloat16)
    emb[:n_tok] = writer.to_bf16("embed", tmpl)
    emb[n_tok:] = emb[2 * C]
    writer.add("model.embed_tokens.weight", emb)
    del emb

    sd = code_dims(spec)
    masks = rng.choice(np.array([-1.0, 1.0], np.float32), size=(L, CODE))
    gamma = bf16_round(SQ / np.sqrt(hd / CODE))
    off = spec.norm_offset
    slots = np.stack([rng.permutation(nk * dk)[:C] for _ in range(L)])
    sc_info = {}
    for l in range(L):
        pre = f"model.layers.{l}."
        last = l == L - 1
        if types[l] == "full_attention":
            q, k, v, o = attention(spec, masks[l], sd, B)
            if scale_check and last:
                sc_info = sc.attention(q, k, v, o, spec, sd, B, f, gamma)
            if spec.attn_gate:
                q = gated_q(spec, q, B)
            writer.add(pre + "self_attn.q_proj.weight", q, exact=not (scale_check and last))
            for name, w in (("k_proj", k), ("v_proj", v), ("o_proj", o)):
                writer.add(pre + f"self_attn.{name}.weight", w)
            writer.add(pre + "self_attn.q_norm.weight", np.full(hd, gamma - off, np.float32))
            writer.add(pre + "self_attn.k_norm.weight", np.full(hd, gamma - off, np.float32))
        else:
            for name, w in gdn_layer(spec, slots[l], codes, B, mc).items():
                suffix = "" if name in ("A_log", "dt_bias") else ".weight"
                writer.add(pre + f"linear_attn.{name}{suffix}", w)
        inter = spec.moe["shared_inter"] if spec.moe else spec.intermediate
        gate = np.zeros((inter, d), np.float32)
        up = np.zeros((inter, d), np.float32)
        down = np.zeros((d, inter), np.float32)
        cu = pointer_units(gate, up, down, B)
        if scale_check and last:
            sc.mlp(gate, up, down, B, f, cu + 1, cu)
        mlp = "mlp.shared_expert." if spec.moe else "mlp."
        for name, w in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            writer.add(pre + f"{mlp}{name}.weight", w)
        if spec.moe:
            moe_block(writer, spec, pre, B)
        writer.add(pre + "input_layernorm.weight", np.full(d, 1.0 - off, np.float32))
        writer.add(pre + "post_attention_layernorm.weight", np.full(d, 1.0 - off, np.float32))
    writer.add("model.norm.weight", np.full(d, 1.0 - off, np.float32))
    extra, s_err = [], None
    if scale_check:
        s_err = n_tok
        extra = [([s_err], sc.lm_row(d, f)[None, :])]
    lm_head(writer, spec, C, codes, extra_rows=extra)

    meta = {
        "program": "hybrid", "classes": C, "fillers": F, "layers": L, "nxt": nxt.tolist(),
        "pools": pool_list, "max_corr": mc, "seed": seed, "layer_types": list(types),
        "layout": {"code": CODE, "KC": KC, "PC": PC, "P": P, "R": R, "GK": GK, "K1": K1, "k1": k1, "B": B,
                   "nb": nb, "bval": BVAL, "rms": RMS, **f},
        "margins": margins, "masks": masks.astype(int).tolist(), "slow_dims": sd,
        "gdn_slots": slots.tolist(),
        "scale_check": {"markers": marker_ids, "s_err": s_err, "window": [sc.LO, sc.HI], **sc_info}
        if scale_check else None,
    }
    return meta, codes
