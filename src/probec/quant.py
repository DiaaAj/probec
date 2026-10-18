"""Quantized formats.

Weights: probec writes quantized checkpoints itself and chooses every scale, so it can pick scales that
reproduce each designed weight bit for bit (powers of two for FP8 and MXFP4, E4M3-representable ones for
NVFP4, a gcd scale and zero point for INT4). The quantizers here return the codes, the scales and whether
the result is exact; `probec certify --quant` reports the last, and the checkpoint writer uses the rest.

Activations: engines compute activation scales at run time from the values' maximum. `fake_quant_act`
reproduces that (quantize, then dequantize), and `fake_quant_linears` applies it to a Hugging Face
model's decoder linears, so the reference model shows what an engine's quantized path should produce.
"""
import contextlib

import torch

E4M3_MAX = 448.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# weight formats: name -> (description, quantizer kwargs)
WEIGHT_FORMATS = {
    "fp8": "FP8 E4M3, one power-of-two scale per tensor",
    "fp8-block": "FP8 E4M3, one power-of-two scale per 128x128 block",
    "int4-awq": "INT4, group 128 along K, fp16 scale and zero point (AWQ)",
    "int4-gptq": "INT4, group 128 along K, fp16 scale, symmetric (GPTQ, zero point 8)",
    "nvfp4": "FP4 E2M1, E4M3 scale per 16 along K",
    "mxfp4": "FP4 E2M1, power-of-two (E8M0) scale per 32 along K",
}
ACT_FORMATS = ("fp8-tensor", "fp8-token", "fp8-group128", "nvfp4", "mxfp4")


def _pow2_ceil(x):
    return torch.exp2(torch.ceil(torch.log2(x)))


def on_e4m3(x):
    """Elementwise: is x (float32 or float64) exactly an E4M3 value?"""
    return x.float().to(torch.float8_e4m3fn).float() == x.float()


def _blocks(w, rows, cols):
    """w [o, i] zero-padded and viewed as [o/rows, rows, i/cols, cols]."""
    o, i = w.shape
    bo, bi = -(-o // rows), -(-i // cols)
    if (bo * rows, bi * cols) != (o, i):
        w = torch.nn.functional.pad(w, (0, bi * cols - i, 0, bo * rows - o))
    return w.reshape(bo, rows, bi, cols)


# --------------------------------------------------------------------------------------------- weights

def quant_fp8(w, block=None):
    """E4M3 weights with power-of-two scales, per tensor (block=None) or per block x block tile.

    Returns (codes as float8_e4m3fn [o, i], scale (scalar or [ceil(o/b), ceil(i/b)], float32), exact).
    Dequantization is codes * scale."""
    w = w.float()
    if block is None:
        a = w.abs().max()
        s = _pow2_ceil(a / E4M3_MAX) if a > 0 else torch.tensor(1.0)
        q = w / s
        return q.to(torch.float8_e4m3fn), s.reshape(()).float(), bool(on_e4m3(q).all())
    t = _blocks(w, block, block)
    a = t.abs().amax(dim=(1, 3))
    s = torch.where(a > 0, _pow2_ceil(a / E4M3_MAX), torch.ones_like(a))
    q = t / s[:, None, :, None]
    exact = bool(on_e4m3(q).all())
    q = q.reshape(t.shape[0] * block, t.shape[2] * block)[: w.shape[0], : w.shape[1]]
    return q.to(torch.float8_e4m3fn), s.float(), exact


def quant_int4(w, group=128, sym=False):
    """4-bit integers per (output row, group of `group` input columns): w = (q - z) * s.

    The scale is the gcd of the group's values (exact for dyadic weights), the zero point z is 8 when
    sym (GPTQ) and otherwise whatever puts the group's minimum at code 0 (AWQ). Returns (codes uint8
    [o, i] in 0..15, scales float16 [o, i/group], zeros uint8 [o, i/group], exact)."""
    o, i = w.shape
    if i % group:
        raise ValueError(f"input dim {i} is not a multiple of the group size {group}")
    t = w.double().reshape(o, i // group, group)
    n = torch.round(t * 2.0 ** 30)
    dyadic = bool(torch.equal(n, t * 2.0 ** 30))
    n = n.long()
    g = n[..., 0].abs()
    for k in range(1, group):
        g = torch.gcd(g, n[..., k])
    g = torch.where(g == 0, torch.full_like(g, 2 ** 30), g)      # all-zero group: scale 1
    lo, hi = (n // g[..., None]).amin(-1), (n // g[..., None]).amax(-1)
    if sym:
        z = torch.full_like(lo, 8)
        fits = (hi <= 7) & (lo >= -8)
    else:
        # a zero point in 0..15 that puts every code in 0..15 (a group without zeros, e.g. all 0.25, gets 0)
        z = (-lo).clamp(0, 15)
        fits = (lo + z >= 0) & (hi + z <= 15)
    s = g.double() / 2.0 ** 30
    q = (n // g[..., None]) + z[..., None]
    scale16 = s.to(torch.float16)
    exact = dyadic and bool(fits.all()) and bool(torch.equal(scale16.double(), s))
    q = q.clamp(0, 15).to(torch.uint8).reshape(o, i)
    return q, scale16, z.to(torch.uint8), exact


def _grid(like):
    return E2M1.to(device=like.device, dtype=like.dtype)


def e2m1_encode(m):
    """Nearest E2M1 code (0..7) for magnitudes m >= 0 (values above 6 saturate); ties go to the lower code."""
    return (m[..., None] - _grid(m)).abs().argmin(-1)


def quant_fp4(w, group, scale_format):
    """E2M1 weights with one scale per `group` input columns. scale_format "e4m3" (NVFP4 block scales,
    used with a global scale of 1) or "e8m0" (MXFP4: powers of two). Picks, per group, the largest grid
    value the group's maximum can map to with a representable scale.

    Returns (sign-magnitude codes uint8 [o, i] in 0..15 (bit 3 = sign), scales float32 [o, i/group],
    exact)."""
    o, i = w.shape
    if i % group:
        raise ValueError(f"input dim {i} is not a multiple of the group size {group}")
    t = w.float().reshape(o, i // group, group)
    a = t.abs().amax(-1)
    best_s = torch.ones_like(a)
    done = a == 0
    for top in (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5):
        s = torch.where(a > 0, a / top, torch.ones_like(a))
        if scale_format == "e8m0":
            s_ok = torch.log2(s) == torch.round(torch.log2(s))
        else:
            s_ok = on_e4m3(s)
        r = t.abs() / s[..., None]
        ok = torch.isin(r, _grid(r)).all(-1) & s_ok & ~done
        best_s = torch.where(ok, s, best_s)
        done |= ok
    r = t.abs() / best_s[..., None]
    codes = e2m1_encode(r) | ((t < 0).to(torch.long) << 3)
    return codes.to(torch.uint8).reshape(o, i), best_s, bool(done.all())


def quant_fp4_modelopt(w, global_scale):
    """NVFP4 with ModelOpt's scale convention: decode global scale weight_scale_2 = amax / (6 * 448) and
    E4M3 block scales amax_block / (6 * weight_scale_2), which lie near 448. The global scale is not a power
    of two, so a weight decodes to within float rounding of its designed value; exact means it rounds back
    to it in bf16. Returns (codes uint8 [o, i], block scales float32 [o, i/16], exact)."""
    o, i = w.shape
    if i % 16:
        raise ValueError(f"input dim {i} is not a multiple of 16")
    t = w.float().reshape(o, i // 16, 16)
    a = t.abs().amax(-1)
    sb = torch.where(a > 0, a / (6.0 * global_scale), torch.ones_like(a))
    sb8 = sb.to(torch.float8_e4m3fn).float()
    s = sb8 * global_scale
    r = t.abs() / s[..., None]
    codes = e2m1_encode(r)
    dq = _grid(r)[codes] * s[..., None] * torch.sign(t)
    exact = bool((dq.to(torch.bfloat16).float() == t).all())
    codes = codes | ((t < 0).to(torch.long) << 3)
    return codes.to(torch.uint8).reshape(o, i), sb8, exact


def quantize_weight(w, fmt):
    """Dispatch by WEIGHT_FORMATS name; returns the quantizer's tuple (its last element is `exact`)."""
    if fmt == "fp8":
        return quant_fp8(w)
    if fmt == "fp8-block":
        return quant_fp8(w, 128)
    if fmt == "int4-awq":
        return quant_int4(w, 128, sym=False)
    if fmt == "int4-gptq":
        return quant_int4(w, 128, sym=True)
    if fmt == "nvfp4":
        return quant_fp4(w, 16, "e4m3")
    if fmt == "mxfp4":
        return quant_fp4(w, 32, "e8m0")
    raise ValueError(f"unknown weight format {fmt!r}; known: {', '.join(WEIGHT_FORMATS)}")


def is_quantized_linear(name, shape):
    """The weights quantized checkpoints quantize: decoder linears and MoE experts, not embeddings, the
    LM head, norms or routers."""
    if len(shape) != 2 or not name.endswith(".weight") or ".layers." not in name:
        return False
    return not name.endswith(("norm.weight", "mlp.gate.weight", "router.weight", "indexer.weights_proj.weight"))


# ----------------------------------------------------------------------------------------- activations

def _fp4_round(r):
    """Round magnitudes r to the E2M1 grid (saturating at 6), keeping the sign of the input."""
    return _grid(r)[e2m1_encode(r.abs())] * torch.sign(r)


def fake_quant_act(x, fmt, global_scale=None):
    """Quantize then dequantize activations x [..., K] the way an engine does at run time, in float32.

    fp8-tensor / fp8-token / fp8-group128: E4M3 with scale amax / 448 per tensor, per row or per 1x128
    group. mxfp4: E2M1 per 32 with the OCP scale 2^(floor(log2 amax) - 2). nvfp4: E2M1 per 16 with block
    scale amax / 6, rounded to E4M3 relative to a static global scale (amax over the tensor if None)."""
    shape = x.shape
    x = x.float().reshape(-1, shape[-1])
    if fmt in ("fp8-tensor", "fp8-token"):
        a = x.abs().amax() if fmt == "fp8-tensor" else x.abs().amax(-1, keepdim=True)
        s = torch.where(a > 0, a / E4M3_MAX, torch.ones_like(a))
        y = (x / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s
    elif fmt == "fp8-group128":
        g = x.reshape(x.shape[0], -1, 128)
        a = g.abs().amax(-1, keepdim=True)
        s = torch.where(a > 0, a / E4M3_MAX, torch.ones_like(a))
        y = ((g / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)
    elif fmt == "mxfp4":
        g = x.reshape(x.shape[0], -1, 32)
        a = g.abs().amax(-1, keepdim=True)
        s = torch.exp2(torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a)))) - 2)
        y = (_fp4_round(g / s) * s).reshape(x.shape)
    elif fmt == "nvfp4":
        gs = (6.0 * E4M3_MAX / x.abs().amax()) if global_scale is None else torch.tensor(float(global_scale), device=x.device)
        g = x.reshape(x.shape[0], -1, 16)
        a = g.abs().amax(-1, keepdim=True)
        sb = (a / 6.0 * gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()       # stored block scale
        s = torch.where(sb > 0, sb / gs, torch.ones_like(sb))
        y = (_fp4_round(g / s) * s).reshape(x.shape)
    else:
        raise ValueError(f"unknown activation format {fmt!r}; known: {', '.join(ACT_FORMATS)}")
    return y.reshape(shape)


@contextlib.contextmanager
def fake_quant_linears(model, act=None, weight=None):
    """Within the block, every quantizable linear of `model` (see is_quantized_linear) runs on
    fake-quantized inputs (`act`, an ACT_FORMATS name) and/or dequantized weights (`weight`, a
    WEIGHT_FORMATS name; the probe's weights are exact, so this is the identity unless they are not)."""
    patched = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or not is_quantized_linear(name + ".weight", mod.weight.shape):
            continue
        w = mod.weight
        if weight is not None:
            w = dequantize(w, weight).to(mod.weight.dtype)

        def forward(x, mod=mod, w=w):
            if act is not None:
                x = fake_quant_act(x, act).to(x.dtype)
            return torch.nn.functional.linear(x, w, mod.bias)

        patched.append((mod, mod.forward))
        mod.forward = forward
    try:
        yield len(patched)
    finally:
        for mod, fwd in patched:
            mod.forward = fwd


def dequantize(w, fmt):
    """The weight an engine reconstructs from probec's checkpoint in format `fmt`."""
    o, i = w.shape
    if fmt in ("fp8", "fp8-block"):
        q, s, _ = quant_fp8(w, None if fmt == "fp8" else 128)
        if s.ndim == 0:
            return q.float() * s
        t = _blocks(q.float(), 128, 128) * s[:, None, :, None]
        return t.reshape(s.shape[0] * 128, s.shape[1] * 128)[:o, :i]
    if fmt in ("int4-awq", "int4-gptq"):
        q, s, z, _ = quant_int4(w, 128, sym=fmt == "int4-gptq")
        g = i // s.shape[1]
        return ((q.reshape(o, -1, g).float() - z[..., None].float()) * s[..., None].float()).reshape(o, i)
    if fmt in ("nvfp4", "mxfp4"):
        q, s, _ = quant_fp4(w, 16 if fmt == "nvfp4" else 32, "e4m3" if fmt == "nvfp4" else "e8m0")
        g = i // s.shape[1]
        mag = E2M1.to(q.device)[(q & 7).long()]
        val = torch.where((q & 8) > 0, -mag, mag)
        return (val.reshape(o, -1, g) * s[..., None]).reshape(o, i)
    raise ValueError(f"unknown weight format {fmt!r}; known: {', '.join(WEIGHT_FORMATS)}")


def certify_weights(tensors, formats, score_only=()):
    """tensors: iterable of (name, tensor). Returns {format: {"tensors": n, "exact": n_exact, "inexact":
    [names], "score_only": {name: max relative error}}}. Score-only tensors (they only shape attention
    scores, which need margin, not exactness) are reported with their rounding error instead."""
    rep = {f: {"tensors": 0, "exact": 0, "inexact": [], "score_only": {}} for f in formats}
    for name, t in tensors:
        if not is_quantized_linear(name, t.shape):
            continue
        for f in formats:
            r = rep[f]
            if name in score_only:
                w = t.float()
                nz = w != 0
                err = ((dequantize(t, f) - w).abs()[nz] / w.abs()[nz]).max() if bool(nz.any()) else torch.tensor(0.0)
                r["score_only"][name] = round(float(err), 4)
                continue
            r["tensors"] += 1
            try:
                exact = quantize_weight(t, f)[-1]
            except ValueError as e:
                exact = False
                name = f"{name} ({e})"
            if exact:
                r["exact"] += 1
            elif len(r["inexact"]) < 10:
                r["inexact"].append(name)
    return rep


def check_quant(model_dir, act=None, weight=None, n_prompts=6, max_len=1200, n_new=16, device="cpu", seed=1):
    """The reference model (HF transformers, bf16) generating under fake quantization, against the oracle:
    {"tokens": n, "mismatches": m, "requests_wrong": r}. Mismatches mean the format breaks the probe's
    exactness (or the probe needs the alignment or pins of docs/QUANTIZED.md), not an engine bug."""
    import numpy as np

    from .check import load
    from .harness import workload
    from .oracle import Probe

    probe = Probe(model_dir)
    model = load(model_dir, "sdpa", torch.bfloat16).to(device)
    prompts = workload(probe, np.random.default_rng(seed), n_prompts, max_len)
    bad = wrong = 0
    with fake_quant_linears(model, act=act, weight=weight), torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p], device=device)
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                 min_new_tokens=n_new, do_sample=False, pad_token_id=probe.pad)[0, len(p):]
            d = sum(int(a != b) for a, b in zip(out.tolist(), probe.expected(p, n_new)))
            bad += d
            wrong += d > 0
    return {"act": act, "weight": weight, "prompts": len(prompts), "tokens": len(prompts) * n_new,
            "mismatches": bad, "requests_wrong": wrong}


def calibrate_input_amax(model_dir, n_prompts=8, max_len=2000, device="cpu", seed=2):
    """Largest |input| of every quantizable linear over the probe's own workload (prefill, HF transformers),
    keyed by the linear's checkpoint weight name. Static activation scales (NVFP4 input_scale) come from
    it, as ModelOpt's calibration would set them. The MoE router's input is recorded too: experts read the
    same normalized hidden state."""
    import numpy as np

    from .check import load
    from .harness import workload
    from .oracle import Probe

    probe = Probe(model_dir)
    model = load(model_dir, "sdpa", torch.bfloat16).to(device)
    amax, hooks = {}, []

    def note(key, x):
        if x.numel():
            amax[key] = max(amax.get(key, 0.0), float(x.detach().abs().max()))

    for name, mod in model.named_modules():
        key = name + ".weight"
        if isinstance(mod, torch.nn.Linear) and (is_quantized_linear(key, mod.weight.shape) or key.endswith("mlp.gate.weight")):
            hooks.append(mod.register_forward_pre_hook(lambda m, inp, key=key: note(key, inp[0])))
        elif hasattr(mod, "gate_up_proj") and hasattr(mod, "act_fn"):
            # transformers' fused experts (3-D weights): recompute each selected expert's inputs
            def experts_hook(m, args, name=name):
                hs, idx = args[0], args[1]
                for e in idx.unique().tolist():
                    if e >= m.gate_up_proj.shape[0]:
                        continue
                    x = hs[(idx == e).any(-1)]
                    gate, up = torch.nn.functional.linear(x, m.gate_up_proj[e]).chunk(2, dim=-1)
                    for proj in ("gate_proj", "up_proj"):
                        note(f"{name}.{e}.{proj}.weight", x)
                    note(f"{name}.{e}.down_proj.weight", m.act_fn(gate) * up)
            hooks.append(mod.register_forward_pre_hook(experts_hook))
    try:
        with torch.no_grad():
            for p in workload(probe, np.random.default_rng(seed), n_prompts, max_len):
                model(torch.tensor([p], device=device))
    finally:
        for h in hooks:
            h.remove()
    return amax
