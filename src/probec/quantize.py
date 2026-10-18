"""probec quantize: a quantized copy of a compiled probe twin, in a checkpoint format engines load.

Every quantized linear (quant.is_quantized_linear) is replaced by the format's tensors and everything else
(embeddings, norms, the LM head, routers) is copied unchanged, as are probe_meta.json and
probe_codes.npy, so the oracle and the harness work on the copy as on the original. The weight scales are
the exact ones from quant.py, so dequantized weights equal the designed ones; the writer refuses a format
in which they would not.
"""
import glob
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

from .quant import is_quantized_linear, quant_fp8

FORMATS = {
    "fp8": {"quant_method": "fp8", "activation_scheme": "dynamic"},
    "fp8-block": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]},
}


def _emit(name, w, fmt):
    """The checkpoint tensors replacing linear weight `name` in format `fmt`, and whether they are exact."""
    base = name[: -len(".weight")]
    if fmt == "fp8":
        q, s, exact = quant_fp8(w)
        return {name: q, base + ".weight_scale": s.reshape(1)}, exact
    if fmt == "fp8-block":
        q, s, exact = quant_fp8(w, 128)
        return {name: q, base + ".weight_scale_inv": s.contiguous()}, exact
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def quantize_twin(src, out, fmt, allow_inexact=False, log=print):
    """Write `src` (a compiled twin) quantized to `fmt` into `out`. Returns a summary dict."""
    if fmt not in FORMATS:
        raise SystemExit(f"probec: unknown format {fmt!r} (have: {', '.join(FORMATS)})")
    if os.path.abspath(src) == os.path.abspath(out):
        raise SystemExit("probec: the output directory must differ from the source twin")
    os.makedirs(out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    n_q = n_kept = 0
    inexact = []
    index = {}
    for fn in shards:
        tensors = {}
        with safe_open(fn, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if is_quantized_linear(name, t.shape):
                    emitted, exact = _emit(name, t, fmt)
                    if not exact:
                        inexact.append(name)
                    tensors.update(emitted)
                    n_q += 1
                else:
                    tensors[name] = t
                    n_kept += 1
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
    cfg["quantization_config"] = dict(FORMATS[fmt])
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
