"""MLA recall: the recall program on DeepSeek-style multi-head latent attention (DeepSeek V2/V3).

The cached latent (kv_lora_rank dims, after kv_a_layernorm) holds the token's masked key code, its
payload code and +1 fill, so every latent entry is ±1: the latent RMS is exactly 1 and the cache (bf16
or FP8) stores it exactly. kv_b_proj expands head 0's no-RoPE key (SK * masked key code) and value
(payload - 1). Queries carry the masked pointer in head 0's no-RoPE part; the RoPE parts are zero, so
content lookups are position-invariant at the model's real rope_theta and YaRN scaling. Engines that
absorb W_UK / W_UV into the query and output (MLA decode) see the same exact values.

MLPs are the recall pointer logic in every layer (the twin sets first_k_dense_replace = num_layers).
"""
import numpy as np
import torch

from ..codes import ballast, make_codes
from . import scalecheck as sc
from .recall import (BVAL, CODE, KC, P, PC, R, RMS, SK, SQ, base_rows, code_meta, lm_head, make_pools,
                     pointer_units, validate)

DEFAULTS = dict(classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24,
                rope_theta=None, seed=0, scale_check=True)


def _marker_code(others, rng, batch=8192, rounds=48, target=24):
    """A ±1 code whose |dot| with every other code is small (best of up to rounds * batch draws)."""
    o = others.astype(np.float32)
    best, best_m = None, None
    for _ in range(rounds):
        cand = rng.choice(np.array([-1.0, 1.0], np.float32), size=(batch, o.shape[1]))
        m = np.abs(cand @ o.T).max(axis=1)
        i = int(m.argmin())
        if best_m is None or m[i] < best_m:
            best, best_m = cand[i], int(m[i])
        if best_m <= target:
            break
    return best.astype(np.int8), best_m


def softmax_scale(spec):
    """qk_head_dim^-0.5, times mscale^2 under YaRN (as DeepSeek and HF apply it)."""
    s = (spec.mla["nope"] + spec.mla["rope"]) ** -0.5
    sc = spec.rope_scaling or {}
    if sc.get("rope_type") == "yarn" and sc.get("mscale_all_dim"):
        f = float(sc.get("factor", 1.0))
        m = 0.1 * float(sc["mscale_all_dim"]) * np.log(f) + 1.0 if f > 1 else 1.0
        s *= m * m
    return float(s)


def _indexer(writer, pre, spec, mk, B, marker=None):
    """DSA indexer (DeepSeek V3.2): head 0 scores the masked pointer against the masked key code, so the
    token the layer looks up is always the indexer's top-1 and lands in the sparse top-k set. The key
    puts the code in the no-RoPE half and its negation in the RoPE half: mean 0 and variance 1, so the
    indexer's LayerNorm is the identity.

    With the scale check (last layer), heads 1 and 2 query +-code(marker A) from a +1 fill dim of the q
    latent. Marker B's code is -code(A), so each marker scores 64 on its own head, while any other key
    gains ReLU(x) + ReLU(-x) = |x| <= the marker code's correlation bound: the markers stay in the
    sparse set the calibrated head attends over."""
    ix = spec.mla["index"]
    d, ql, ihd, rope = spec.hidden, spec.mla["q_lora"], ix["head_dim"], spec.mla["rope"]
    if ql is None or ihd - rope < CODE or rope < CODE:
        raise SystemExit("probec: DSA indexer needs q_lora_rank and >= 64 dims in each half of index_head_dim")
    wq = np.zeros((ix["heads"] * ihd, ql), np.float32)
    wk = np.zeros((ihd, d), np.float32)
    for t in range(CODE):
        wq[rope + t, t] = mk[t]                         # head 0 no-RoPE part: mask (.) pointer (q latent dims)
        wk[rope + t, KC + t] = RMS * mk[t]              # no-RoPE part: mask (.) key code
        wk[t, KC + t] = -RMS * mk[t]                    # RoPE part: the negation (zero mean)
    wp = np.zeros((ix["heads"], d), np.float32)
    wp[0, B] = 0.5                                      # head 0 weight 1
    if marker is not None:
        for h, sgn in ((1, 1.0), (2, -1.0)):
            for t in range(CODE):
                wq[h * ihd + rope + t, CODE] = sgn * mk[t] * marker[t]
            wp[h, B] = 0.5
    writer.add(pre + "indexer.wq_b.weight", wq)
    writer.add(pre + "indexer.wk.weight", wk)
    writer.add(pre + "indexer.k_norm.weight", np.ones(ihd, np.float32))
    writer.add(pre + "indexer.k_norm.bias", np.zeros(ihd, np.float32))
    writer.add(pre + "indexer.weights_proj.weight", wp)


def build(spec, writer, classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512",
          max_corr=24, rope_theta=None, seed=0, scale_check=True):
    if not spec.mla:
        raise SystemExit("probec: the mla program needs a DeepSeek MLA architecture")
    m = spec.mla
    dsa = bool(m.get("index"))
    scale_check = bool(scale_check) and spec.heads >= 2 and (not dsa or m["index"]["heads"] >= 3)
    n_sc = sc.N_DIMS_PM if scale_check else 0
    validate(spec, classes, fillers, n_pm=4 * CODE + n_sc, units=2 * CODE + 1 + sc.UNITS, extra_vocab=3)
    nope, rope, vh, kvl, ql = m["nope"], m["rope"], m["v_head"], m["kv_lora"], m["q_lora"]
    if nope < CODE or vh < CODE or kvl < 2 * CODE + 3 or (ql is not None and ql < CODE + 1):
        raise SystemExit("probec: MLA dims too small for 64-dim codes")
    rng = np.random.default_rng(seed)
    C, F, L, d = classes, fillers, spec.layers, spec.hidden
    nh = spec.heads
    qk = nope + rope
    k1, nb = ballast(4 * CODE + n_sc, d, BVAL, RMS)
    K1 = 4 * CODE + n_sc
    B = K1 + k1
    f = {"MA": 4 * CODE, "MB": 4 * CODE + 1, "FLAG": 4 * CODE + 2, "CAL": B + nb} if scale_check else {}
    n_tok = 2 * C + F + (2 if scale_check else 0)
    codes = make_codes(n_tok, CODE, max_corr, rng)
    mark_corr = None
    if scale_check and dsa:                            # sparse attention: marker codes A and -A (see _indexer)
        a, mark_corr = _marker_code(codes[:n_tok - 2], np.random.default_rng(seed + 7919))
        codes[n_tok - 2], codes[n_tok - 1] = a, -a
    nxt, pool_list = make_pools(C, pools, rng)

    tmpl = np.zeros((n_tok, d), np.float32)
    tmpl[:2 * C + F] = base_rows(C, F, codes, nxt, d, B, nb, K1, k1)
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
    emb = torch.empty((spec.vocab, d), dtype=torch.bfloat16)
    emb[:n_tok] = writer.to_bf16("embed", tmpl)
    emb[n_tok:] = emb[2 * C]
    writer.add("model.embed_tokens.weight", emb)
    del emb

    LK, LP = 0, CODE                                   # latent: masked key code, payload code, markers, +1 fill
    LM, LAB = 2 * CODE, 2 * CODE + 1                   # scale check: is-a-marker, A-vs-B (both +-1)
    LC = 2 * CODE + 2                                  # first fill dim (constant +1)
    scale = softmax_scale(spec)
    masks = rng.choice(np.array([-1.0, 1.0], np.float32), size=(L, CODE))
    sc_info = {}
    for l in range(L):
        mk = masks[l]
        last = scale_check and l == L - 1
        pre = f"model.layers.{l}.self_attn."
        kva = np.zeros((kvl + rope, d), np.float32)    # latent rows, then k_rot rows (all zero)
        for t in range(CODE):
            kva[LK + t, KC + t] = RMS * mk[t]          # latent key = mask_l (.) key code
            kva[LP + t, PC + t] = RMS                  # latent payload
        if scale_check:                                # MA, MB: (+1,-1) marker A, (-1,+1) B, (-1,-1) others
            kva[LM, f["MA"]] = RMS                     # LM = MA + MB + 1: +1 on both markers, -1 elsewhere
            kva[LM, f["MB"]] = RMS
            kva[LM, B] = 0.5
            kva[LAB, f["MB"]] = -RMS                   # LAB = -MB: -1 on marker B only
        else:
            kva[LM:LC, B] = 0.5
        kva[LC:kvl, B] = 0.5                           # fill: +1
        kvb = np.zeros((nh * (nope + vh), kvl), np.float32)
        for t in range(CODE):
            kvb[t, LK + t] = SK                        # head 0 k_nope = SK * latent key
            kvb[nope + t, LP + t] = 1.0                # head 0 v = payload - 1
            kvb[nope + t, LC] = -1.0
        qnope1 = None
        if last:                                       # head 1: calibrated two-marker lookup (scale check)
            # k_0 = LAB carries A vs B alone, so the absorbed query (W_UK^T q, which FP8 MLA kernels
            # quantize) keeps c in its own latent dim instead of riding on the large marker term;
            # k_i = 2 LM + 2 (4 on markers, 0 elsewhere) for i >= 1; every absorbed entry is <= 252
            h1 = nope + vh
            c = float(np.log(sc.RATIO) / (2.0 * scale))   # score_A - score_B = 2c = ln 4 / scale
            qnope1 = np.full(CODE, 2.0, np.float32)
            qnope1[0] = c
            kvb[h1 + 0, LAB] = 1.0
            for i in range(1, CODE):
                kvb[h1 + i, LM] = 2.0
                kvb[h1 + i, LC] = 2.0
            kvb[h1 + nope + 0, LAB] = 1.0              # v = LAB: +1 marker A, -1 marker B -> 0.8 - 0.2
            sc_info = {"q_flip": c, "q_head": 1, "softmax_scale": scale}
        if ql is None:
            qp = np.zeros((nh * qk, d), np.float32)
            for t in range(CODE):
                qp[t, P + t] = RMS * SQ * mk[t]        # head 0 q_nope = SQ * mask_l (.) pointer
            if qnope1 is not None:
                qp[qk:qk + CODE, B] = qnope1 / 2.0
            writer.add(pre + "q_proj.weight", qp, exact=qnope1 is None)
        else:
            qa = np.zeros((ql, d), np.float32)
            for t in range(CODE):
                qa[t, P + t] = RMS                     # q latent: pointer, then +1 fill (RMS exactly 1)
            qa[CODE:ql, B] = 0.5
            qb = np.zeros((nh * qk, ql), np.float32)
            for t in range(CODE):
                qb[t, t] = SQ * mk[t]
            if qnope1 is not None:
                qb[qk:qk + CODE, CODE] = qnope1        # from a +1 fill dim of the q latent
            writer.add(pre + "q_a_proj.weight", qa)
            writer.add(pre + "q_a_layernorm.weight", np.ones(ql, np.float32))
            writer.add(pre + "q_b_proj.weight", qb, exact=qnope1 is None)
        o = np.zeros((d, nh * vh), np.float32)
        for t in range(CODE):
            o[R + t, t] = 1.0                          # register += v
        if last:
            o[f["CAL"], vh + 0] = 1.0                  # head 1 output -> CAL
        if dsa:
            _indexer(writer, pre, spec, mk, B, marker=codes[n_tok - 2] if last else None)
        writer.add(pre + "kv_a_proj_with_mqa.weight", kva)
        writer.add(pre + "kv_a_layernorm.weight", np.ones(kvl, np.float32))
        writer.add(pre + "kv_b_proj.weight", kvb)
        writer.add(pre + "o_proj.weight", o)
        inter = spec.intermediate
        gate = np.zeros((inter, d), np.float32)
        up = np.zeros((inter, d), np.float32)
        down = np.zeros((d, inter), np.float32)
        cu = pointer_units(gate, up, down, B)
        if last:
            sc.mlp(gate, up, down, B, f, cu + 1, cu)
        lp = f"model.layers.{l}."
        writer.add(lp + "mlp.gate_proj.weight", gate)
        writer.add(lp + "mlp.up_proj.weight", up)
        writer.add(lp + "mlp.down_proj.weight", down)
        writer.add(lp + "input_layernorm.weight", np.ones(d, np.float32))
        writer.add(lp + "post_attention_layernorm.weight", np.ones(d, np.float32))
    writer.add("model.norm.weight", np.ones(d, np.float32))
    extra, s_err = [], None
    if scale_check:
        s_err = n_tok
        extra = [([s_err], sc.lm_row(d, f)[None, :])]
    lm_head(writer, spec, C, codes, extra_rows=extra)

    mc, margins = code_meta(codes, qk)
    margins["score_nats"] = (CODE - mc) * SQ * SK * scale
    if dsa:                                            # indexer: read token (and markers) vs any other key
        margins["index_dot"] = CODE - mc - (mark_corr or 0)
        if margins["index_dot"] < 8:
            raise SystemExit(f"probec: DSA indexer margin too small ({margins['index_dot']})")
    meta = {
        "program": "mla", "classes": C, "fillers": F, "layers": L, "nxt": nxt.tolist(),
        "pools": pool_list, "max_corr": mc, "seed": seed, "rope_theta": spec.rope_theta,
        "layout": {"code": CODE, "KC": KC, "PC": PC, "P": P, "R": R, "K1": K1, "k1": k1, "B": B,
                   "nb": nb, "bval": BVAL, "rms": RMS, **f,
                   "latent": {"LK": LK, "LP": LP, "LC": LC, "kv_lora": kvl}},
        "margins": margins, "masks": masks.astype(int).tolist(), "softmax_scale": scale,
        "scale_check": {"markers": marker_ids, "s_err": s_err, "window": [sc.LO, sc.HI], **sc_info}
        if scale_check else None,
    }
    return meta, codes
