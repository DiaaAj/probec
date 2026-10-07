"""MoE recall: the recall program on a sparse MoE architecture (Qwen3-MoE), with expert tagging.

Attention is the recall program's. The MLP of every layer is an MoE block:
- Routing is decided by a ROUTE field in each token's embedding: ±1 per expert, +1 on a fixed random
  k-subset. Router logits are ±16 (margin 32), so top-k picks exactly that subset and the renormalized
  weights are exactly 1/k.
- Every expert carries the recall pointer logic, so the k weighted outputs sum to exactly one pointer
  update. A dropped or duplicated expert, or wrong weights, breaks the pointer.
- Expert e also flips its own TAG bit (TAG := 1, then -2 from expert e). The next layer compares TAG with
  -ROUTE: a mismatch on expert j adds 1 to DIAG[j] and corrupts the token's key code (so a misrouted
  prompt token can no longer be found). Diagnostic tokens D_j (logit 100 * DIAG[j]) then name the expert.
  The last layer's routing is unchecked (no layer after it reads TAG).
- The softmax-scale check of the recall program runs in the last layer (its MLP units live in every
  expert, so the weighted sum is exact).
"""
import numpy as np
import torch

from ..codes import ballast, make_codes, slow_dims
from . import scalecheck as sc
from .recall import (BVAL, CODE, G, KC, P, R, RMS, SQ, attention, attn_extras, base_rows, bf16_round,
                     code_meta, lm_head, make_pools, pointer_units, validate)

ROUTE_LOGIT = 16.0              # router logits +-16
ERR_K = 16.0                    # err units: silu(+-16 * (TAG + ROUTE)) in {0, 32}
DIAG_LOGIT = 100.0
KEY_CORRUPT = 8.0

DEFAULTS = dict(classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24,
                rope_theta=1e12, seed=0, scale_check=True)


def is_moe_layer(spec, l):
    m = spec.moe
    return l not in m["mlp_only"] and (l + 1) % m["sparse_step"] == 0


def build(spec, writer, classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512",
          max_corr=24, rope_theta=1e12, seed=0, scale_check=True):
    if not spec.moe:
        raise SystemExit("probec: the moe program needs an MoE architecture (e.g. Qwen3MoeForCausalLM)")
    m = spec.moe
    E, K, I = m["experts"], m["top_k"], m["inter"]
    scale_check = bool(scale_check) and sc.supported(spec)
    n_sc = sc.N_DIMS_PM if scale_check else 0
    units = 2 * CODE + 1 + 3 * E + (sc.UNITS if scale_check else 0)
    n_pm = 4 * CODE + 2 * E + n_sc
    validate(spec, classes, fillers, n_pm=n_pm, units=2 * CODE + 1 + sc.UNITS, extra_vocab=E + 3)
    if I < units:
        raise SystemExit(f"probec: moe_intermediate_size {I} < {units} units needed")
    rng = np.random.default_rng(seed)
    C, F, L, d, hd = classes, fillers, spec.layers, spec.hidden, spec.head_dim
    ROUTE, TAG = 4 * CODE, 4 * CODE + E
    SC0 = 4 * CODE + 2 * E
    k1, nb = ballast(n_pm, d, BVAL, RMS)
    K1 = SC0 + n_sc
    B = K1 + k1
    DIAG = B + nb
    f = {"MA": SC0, "MB": SC0 + 1, "FLAG": SC0 + 2, "CAL": DIAG + E} if scale_check else {}
    if DIAG + E + (1 if scale_check else 0) > d:
        raise SystemExit(f"probec: hidden size {d} too small for {E} experts")
    n_tok = 2 * C + F + (2 if scale_check else 0)
    codes = make_codes(n_tok, CODE, max_corr, rng)
    nxt, pool_list = make_pools(C, pools, rng)

    routes = np.stack([np.sort(rng.choice(E, K, replace=False)) for _ in range(n_tok)])
    route_pm = -np.ones((n_tok, E), np.float32)
    route_pm[np.arange(n_tok)[:, None], routes] = 1.0
    tmpl = np.zeros((n_tok, d), np.float32)
    tmpl[:2 * C + F] = base_rows(C, F, codes, nxt, d, B, nb, K1, k1)
    marker_ids = []
    if scale_check:
        marker_ids = [2 * C + F, 2 * C + F + 1]
        for t in marker_ids:
            tmpl[t, KC:KC + CODE] = tmpl[t, 64:64 + CODE] = tmpl[t, P:P + CODE] = codes[t]
            tmpl[t, R:R + CODE] = 1.0
            tmpl[t, B:B + nb] = BVAL
            if k1:
                tmpl[t, K1:K1 + k1] = 1.0
        sc.embed(tmpl, f, np.arange(2 * C + F), [marker_ids[0]], [marker_ids[1]])
    tmpl[:, ROUTE:ROUTE + E] = route_pm
    tmpl[:, TAG:TAG + E] = -route_pm                   # as if the previous layer routed correctly
    emb = torch.empty((spec.vocab, d), dtype=torch.bfloat16)
    emb[:n_tok] = writer.to_bf16("embed", tmpl)
    emb[n_tok:] = emb[2 * C]
    writer.add("model.embed_tokens.weight", emb)
    del emb

    sd = slow_dims(hd, CODE, spec.neox)
    masks = rng.choice(np.array([-1.0, 1.0], np.float32), size=(L, CODE))
    gamma = bf16_round(SQ / np.sqrt(hd / CODE))

    def common_expert(with_scale):
        """Units shared by every expert. The k routed experts are weighted 1/k each, so shared logic
        sums back to exactly one copy; only expert-specific outputs are scaled by k."""
        gate = np.zeros((I, d), np.float32)
        up = np.zeros((I, d), np.float32)
        down = np.zeros((d, I), np.float32)
        cu = pointer_units(gate, up, down, B, 0)
        u = cu + 1
        for j in range(E):                               # err_j = |TAG_j + ROUTE_j| / 2 in {0, 1}
            for sgn in (1.0, -1.0):
                gate[u, TAG + j] = sgn * RMS * ERR_K
                gate[u, ROUTE + j] = sgn * RMS * ERR_K
                up[u, B] = 0.5                           # up = 1
                down[DIAG + j, u] = 1 / (2 * ERR_K)      # the active unit is 2*ERR_K -> DIAG += 1
                down[KC:KC + CODE, u] = KEY_CORRUPT / (2 * ERR_K)
                u += 1
        for j in range(E):                               # TAG reset: TAG += 1 - TAG
            gate[u, B] = G / 2
            up[u, TAG + j] = RMS
            down[TAG + j, u] = -1 / G
            u += 1
        down[TAG:TAG + E, cu] += 1 / G
        if with_scale:
            sc.mlp(gate, up, down, B, f, u, cu)
        return gate, up, down, cu

    sc_info = {}
    for l in range(L):
        last = l == L - 1
        q, k, v, o = attention(spec, masks[l], sd, B)
        if scale_check and last:
            sc_info = sc.attention(q, k, v, o, spec, sd, B, f, gamma)
        pre = f"model.layers.{l}."
        writer.add(pre + "self_attn.q_proj.weight", q, exact=not (scale_check and last))
        for name, w in (("self_attn.k_proj", k), ("self_attn.v_proj", v), ("self_attn.o_proj", o)):
            writer.add(pre + name + ".weight", w)
        attn_extras(writer, spec, pre)
        if is_moe_layer(spec, l):
            gate_c, up_c, down_c, const_unit = common_expert(scale_check and last)
            router = np.zeros((E, d), np.float32)
            router[np.arange(E), ROUTE + np.arange(E)] = RMS * ROUTE_LOGIT
            writer.add(pre + "mlp.gate.weight", router)
            for e in range(E):
                down = down_c.copy()
                down[TAG + e, const_unit] += -2.0 * K / G    # expert e flips TAG[e] (-2 after weight 1/k)
                writer.add(pre + f"mlp.experts.{e}.gate_proj.weight", gate_c)
                writer.add(pre + f"mlp.experts.{e}.up_proj.weight", up_c)
                writer.add(pre + f"mlp.experts.{e}.down_proj.weight", down)
        else:
            inter = spec.intermediate
            gate = np.zeros((inter, d), np.float32)
            up = np.zeros((inter, d), np.float32)
            down = np.zeros((d, inter), np.float32)
            cu = pointer_units(gate, up, down, B)
            if scale_check and last:
                sc.mlp(gate, up, down, B, f, cu + 1, cu)
            writer.add(pre + "mlp.gate_proj.weight", gate)
            writer.add(pre + "mlp.up_proj.weight", up)
            writer.add(pre + "mlp.down_proj.weight", down)
        writer.add(pre + "input_layernorm.weight", np.ones(d, np.float32))
        writer.add(pre + "post_attention_layernorm.weight", np.ones(d, np.float32))
    writer.add("model.norm.weight", np.ones(d, np.float32))
    diag_ids = list(range(n_tok, n_tok + E))
    diag_rows = np.zeros((E, d), np.float32)
    diag_rows[np.arange(E), DIAG + np.arange(E)] = RMS * DIAG_LOGIT
    extra = [(diag_ids, diag_rows)]
    s_err = None
    if scale_check:
        s_err = n_tok + E
        extra.append(([s_err], sc.lm_row(d, f)[None, :]))
    lm_head(writer, spec, C, codes, extra_rows=extra)

    mc, margins = code_meta(codes, hd)
    margins["router_nats"] = 2 * ROUTE_LOGIT
    meta = {
        "program": "moe", "classes": C, "fillers": F, "layers": L, "nxt": nxt.tolist(),
        "pools": pool_list, "max_corr": mc, "seed": seed, "rope_theta": rope_theta,
        "layout": {"code": CODE, "KC": KC, "PC": 64, "P": P, "R": R, "ROUTE": ROUTE, "TAG": TAG, "E": E,
                   "K1": K1, "k1": k1, "B": B, "nb": nb, "DIAG": DIAG, "bval": BVAL, "rms": RMS, **f},
        "margins": margins, "masks": masks.astype(int).tolist(), "slow_dims": sd,
        "experts": E, "top_k": K, "routes": routes.tolist(), "diag_base": n_tok,
        "moe_layers": [l for l in range(L) if is_moe_layer(spec, l)],
        "scale_check": {"markers": marker_ids, "s_err": s_err, "window": [sc.LO, sc.HI], **sc_info}
        if scale_check else None,
    }
    return meta, codes
