"""probec quantize: a quantized copy of a compiled probe twin, in a checkpoint format engines load.

Every quantized linear (quant.is_quantized_linear) is replaced by the format's tensors and everything else
(embeddings, norms, the LM head, MoE routers) is copied unchanged, as are probe_meta.json and
probe_codes.npy, so the oracle and the harness work on the copy as on the original. The weight scales are
the exact ones from quant.py, so dequantized weights equal the designed ones; the writer refuses a format
in which they would not be. Layouts follow what vLLM 0.31 and SGLang 0.5.22 load (docs/QUANTIZED.md):

- fp8: E4M3 `weight` [N, K] and a [1] F32 `weight_scale`, one power-of-two scale shared by the shards an
  engine fuses (q/k/v, gate/up), because engines requantize fused shards to their largest scale;
- fp8-block: E4M3 `weight` and F32 `weight_scale_inv` [ceil(N/128), ceil(K/128)], a multiplier despite
  its name, powers of two (what DeepGEMM's UE8M0 requantization would produce anyway);
- int4-awq: `qweight` I32 [K, N/8] packed along N in AWQ's [0, 2, 4, 6, 1, 3, 5, 7] order, `qzeros` I32
  [K/128, N/8] packed the same way, `scales` F16 [K/128, N]; W = (q - z) * s;
- int4-gptq: symmetric, `qweight` I32 [K/8, N] packed along K, `qzeros` 0x77777777, `scales` F16
  [K/128, N], `g_idx` k // 128; W = (q - 8) * s;
- nvfp4 (ModelOpt): `weight` U8 [N, K/2] with the even element in the low nibble, E4M3 `weight_scale`
  [N, K/16] (not swizzled), `weight_scale_2` = 1 and a static `input_scale` = calibrated amax / (6 * 448),
  identical across fused shards and experts.
"""
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .quant import E4M3_MAX, _pow2_ceil, is_quantized_linear, on_e4m3, quant_fp4, quant_fp8, quant_int4

FORMATS = {
    "fp8": {"quant_method": "fp8", "activation_scheme": "dynamic"},
    "fp8-block": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]},
    "int4-awq": {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "gemm",
                 "modules_to_not_convert": None},
    "int4-gptq": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
                  "lm_head": False},
    "nvfp4": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16},
}
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def fused_group(name):
    """The fused linear a shard belongs to in vLLM and SGLang (q/k/v, gate/up, MLA's q_a/kv_a), or itself."""
    for shards, fused in ((("q_proj", "k_proj", "v_proj"), "qkv_proj"), (("gate_proj", "up_proj"), "gate_up_proj"),
                          (("q_a_proj", "kv_a_proj_with_mqa"), "fused_qkv_a_proj")):
        for s in shards:
            if name.endswith(f".{s}.weight"):
                return name[: -len(f"{s}.weight")] + fused
    return name


def pack_nibbles(q, dim, order=None):
    """Pack 4-bit codes (0..15) eight per int32 along `dim`: nibble j holds element 8p + order[j]."""
    q = q.to(torch.int64).movedim(dim, -1)
    *lead, n = q.shape
    if n % 8:
        raise ValueError(f"cannot pack {n} values eight per int32")
    g = q.reshape(*lead, n // 8, 8)
    if order is not None:
        g = g[..., list(order)]
    v = (g << (4 * torch.arange(8))).sum(-1)
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
    return v.movedim(-1, dim)


def unpack_nibbles(v, dim, order=None):
    """Inverse of pack_nibbles."""
    v = v.to(torch.int64).movedim(dim, -1) & 0xFFFFFFFF
    nib = (v[..., None] >> (4 * torch.arange(8))) & 0xF
    if order is not None:
        inv = [0] * 8
        for j, c in enumerate(order):
            inv[c] = j
        nib = nib[..., inv]
    return nib.reshape(*v.shape[:-1], -1).movedim(-1, dim)


def _emit(name, w, fmt, scale=None, input_amax=None):
    """The checkpoint tensors replacing linear weight `name`, and whether they are exact."""
    base = name[: -len(".weight")]
    if fmt == "fp8":
        if scale is None:
            raise ValueError(f"{name}: fp8 needs the fused group's scale")
        q = w.float() / scale
        exact = bool(on_e4m3(q).all())
        # one tensor per shard: safetensors refuses tensors that share memory
        return {name: q.to(torch.float8_e4m3fn), base + ".weight_scale": scale.reshape(1).float().clone()}, exact
    if fmt == "fp8-block":
        q, s, exact = quant_fp8(w, 128)
        return {name: q, base + ".weight_scale_inv": s.contiguous()}, exact
    if fmt in ("int4-awq", "int4-gptq"):
        sym = fmt == "int4-gptq"
        q, s, z, exact = quant_int4(w, 128, sym=sym)
        k = w.shape[1]
        if sym:
            out = {base + ".qweight": pack_nibbles(q.t(), 0),
                   base + ".qzeros": torch.full((k // 128, w.shape[0] // 8), 0x77777777, dtype=torch.int32),
                   base + ".scales": s.t().contiguous(),
                   base + ".g_idx": (torch.arange(k) // 128).to(torch.int32)}
        else:
            out = {base + ".qweight": pack_nibbles(q.t(), 1, AWQ_ORDER),
                   base + ".qzeros": pack_nibbles(z.t(), 1, AWQ_ORDER),
                   base + ".scales": s.t().contiguous()}
        return out, exact
    if fmt == "nvfp4":
        codes, s, exact = quant_fp4(w, 16, "e4m3")
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        if input_amax is None:
            raise ValueError(f"{name}: nvfp4 needs a calibrated input amax (--calibrate)")
        return {name: packed, base + ".weight_scale": s.to(torch.float8_e4m3fn),
                base + ".weight_scale_2": torch.ones(1),
                base + ".input_scale": torch.tensor([input_amax / (6.0 * E4M3_MAX)])}, exact
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def _ignore_list(names):
    """Modules engines must leave in bf16, by full name: the LM head and every MoE router (vLLM would
    otherwise quantize the router; SGLang never does)."""
    return ["lm_head"] + sorted(n[: -len(".weight")] for n in names if n.endswith("mlp.gate.weight"))


def quantize_twin(src, out, fmt, allow_inexact=False, input_amax=None, log=print):
    """Write `src` (a compiled twin) quantized to `fmt` into `out`. input_amax: {weight name: max |input|}
    from quant.calibrate_input_amax, needed for nvfp4. Returns a summary dict."""
    if fmt not in FORMATS:
        raise SystemExit(f"probec: unknown format {fmt!r} (have: {', '.join(FORMATS)})")
    if os.path.abspath(src) == os.path.abspath(out):
        raise SystemExit("probec: the output directory must differ from the source twin")
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    names, scales = [], {}
    if fmt in ("fp8", "nvfp4"):                         # first pass: what fused shards must share
        group_amax = {}
        for fn in shards:
            with safe_open(fn, framework="pt") as f:
                for name in f.keys():
                    names.append(name)
                    t = f.get_tensor(name)
                    if is_quantized_linear(name, t.shape):
                        g = fused_group(name)
                        a = float(t.float().abs().max()) if fmt == "fp8" else (input_amax or {}).get(name)
                        if a is not None:
                            group_amax[g] = max(group_amax.get(g, 0.0), a)
        if fmt == "fp8":
            scales = {g: _pow2_ceil(torch.tensor(a / E4M3_MAX)) if a > 0 else torch.tensor(1.0)
                      for g, a in group_amax.items()}
        else:
            scales = dict(group_amax)
            # experts share one activation scale per layer and kind (gate/up, down): both engines collapse
            # the experts' input scales to their maximum
            kind_max = {}
            for g, a in group_amax.items():
                if ".mlp.experts." in g:
                    k = (g.split(".mlp.experts.")[0], g.rsplit(".", 1)[1])
                    kind_max[k] = max(kind_max.get(k, 0.0), a)
            for g in scales:
                if ".mlp.experts." in g:
                    scales[g] = kind_max[(g.split(".mlp.experts.")[0], g.rsplit(".", 1)[1])]
    else:
        for fn in shards:
            with safe_open(fn, framework="pt") as f:
                names += list(f.keys())
    os.makedirs(out, exist_ok=True)
    n_q = n_kept = 0
    inexact, index = [], {}
    for fn in shards:
        tensors = {}
        with safe_open(fn, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if not is_quantized_linear(name, t.shape):
                    tensors[name] = t
                    n_kept += 1
                    continue
                g = fused_group(name)
                emitted, exact = _emit(name, t, fmt, scale=scales.get(g) if fmt == "fp8" else None,
                                       input_amax=scales.get(g) if fmt == "nvfp4" else None)
                if not exact:
                    inexact.append(name)
                tensors.update(emitted)
                n_q += 1
        if inexact and not allow_inexact:
            raise SystemExit(f"probec: {len(inexact)} weights are not exact in {fmt} (e.g. {inexact[0]}); "
                             "a twin compiled with --no-scale-check usually is")
        base = os.path.basename(fn)
        save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, base), metadata={"format": "pt"})
        index.update({k: base for k in tensors})
    if len(shards) > 1:
        total = sum(os.path.getsize(os.path.join(out, os.path.basename(f))) for f in shards)
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": index}, f, indent=1)
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, name))
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    qc = dict(FORMATS[fmt])
    ignore = _ignore_list(names)
    if fmt.startswith("fp8"):
        qc["ignored_layers"] = qc["modules_to_not_convert"] = ignore
    elif fmt == "nvfp4":
        qc["ignore"] = ignore
    cfg["quantization_config"] = qc
    cfg.setdefault("torch_dtype", "bfloat16")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "probe_meta.json")) as f:
        meta = json.load(f)
    meta["quantized"] = {"format": fmt, "source": os.path.abspath(src), "linears": n_q, "inexact": inexact}
    with open(os.path.join(out, "probe_meta.json"), "w") as f:
        json.dump(meta, f)
    rep = {"format": fmt, "quantized_linears": n_q, "other_tensors": n_kept, "inexact": inexact, "out": out}
    log(json.dumps(rep))
    return rep
