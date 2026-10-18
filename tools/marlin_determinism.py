"""Kernel-level check of SGLang's FP8 Marlin GEMM (weight-only FP8, used below SM89) on the real FP8-block
weights of a probec checkpoint. The weights and inputs are dyadic, so every partial sum is exact in fp32
and any correct kernel returns the exact product. Three phases:
  1. sweep: every M in 1..1024 (and some larger), one call each, against the exact product;
  2. repeats: selected M, --reps calls each: same bits every time?
  3. two streams: the same layer (one lock workspace) launched on two streams at once;
  4. stream ordering: input produced, GEMM run and output read on one side stream (SGLang's overlap mode).

    python tools/marlin_determinism.py <fp8-block twin dir> [--layers 0,27] [--reps 20]
Prints one line per failing case, then summary lines starting with MARLIN.
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open


def load_tensors(d):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def marlin_layer(w, s, block):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    n, k = w.shape
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = block
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s.cuda().float(), requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--rep-ms", default="1,8,16,17,64,100,128,256,512,1000,2048,4096")
    a = ap.parse_args()
    from sglang.srt.layers.quantization.marlin_utils_fp8 import apply_fp8_marlin_linear
    with open(os.path.join(a.twin, "config.json")) as f:
        block = json.load(f)["quantization_config"].get("weight_block_size")
    t = load_tensors(a.twin)
    g = torch.Generator(device="cuda").manual_seed(0)
    grid = torch.tensor([0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 0.125, -0.125], device="cuda")
    sweep_ms = list(range(1, 1025)) + [1100, 1500, 2000, 2047, 2048, 2049, 3000, 4096, 8192]
    stats = {"sweep": [0, 0], "repeat": [0, 0, 0], "streams": [0, 0], "order": [0, 0]}

    def rand_x(m, k):
        return grid[torch.randint(0, len(grid), (m, k), generator=g, device="cuda")].to(torch.bfloat16)

    for l in [int(x) for x in a.layers.split(",")]:
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
            name = f"model.layers.{l}.{proj}"
            w, s = t[name + ".weight"], t[name + ".weight_scale_inv"]
            ref_w = (w.float().view(w.shape[0] // block[0], block[0], w.shape[1] // block[1], block[1])
                     * s.float()[:, None, :, None]).view(w.shape).cuda()
            layer = marlin_layer(w, s, block)
            n, k = w.shape

            def run(x):
                return apply_fp8_marlin_linear(x, layer.weight, layer.weight_scale, layer.workspace, n, k, None)

            bad_ms = []
            for m in sweep_ms:
                x = rand_x(m, k)
                out = run(x)
                ref = (x.float() @ ref_w.T).to(torch.bfloat16)
                stats["sweep"][0] += 1
                if not torch.equal(out, ref):
                    stats["sweep"][1] += 1
                    bad_ms.append((m, int((out != ref).any(1).sum())))
            if bad_ms:
                print(f"{name} N={n} K={k} sweep: {len(bad_ms)} of {len(sweep_ms)} M values wrong; "
                      f"(M, wrong rows): {bad_ms[:20]}", flush=True)
            for m in [int(x) for x in a.rep_ms.split(",")]:
                x = rand_x(m, k)
                ref = (x.float() @ ref_w.T).to(torch.bfloat16)
                outs = [run(x).clone() for _ in range(a.reps)]
                torch.cuda.synchronize()
                stats["repeat"][0] += 1
                differ = sum(int(not torch.equal(o, outs[0])) for o in outs[1:])
                wrong = sum(int(not torch.equal(o, ref)) for o in outs)
                stats["repeat"][1] += int(differ > 0)
                stats["repeat"][2] += int(wrong > 0)
                if differ or wrong:
                    print(f"{name} N={n} K={k} M={m} repeats: {differ}/{a.reps - 1} differ from the first, "
                          f"{wrong}/{a.reps} differ from the exact product", flush=True)
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            for m in (64, 512, 2048):
                x = rand_x(m, k)
                ref = (x.float() @ ref_w.T).to(torch.bfloat16)
                torch.cuda.synchronize()
                outs = []
                for r in range(a.reps):
                    with torch.cuda.stream(streams[r % 2]):
                        outs.append(run(x))
                torch.cuda.synchronize()
                stats["streams"][0] += 1
                wrong = sum(int(not torch.equal(o, ref)) for o in outs)
                if wrong:
                    stats["streams"][1] += 1
                    print(f"{name} N={n} K={k} M={m} two streams: {wrong}/{a.reps} differ from the exact product",
                          flush=True)
            # 4. stream ordering, as in SGLang's overlap mode (forward on a side stream): on one side stream,
            # produce the input behind ~30 ms of other work, run the GEMM, copy its output. A kernel launched on
            # another stream than torch's current one reads the input before it exists, or is read too early.
            side = torch.cuda.Stream()
            busy = torch.randn((8192, 8192), device="cuda")
            for m in (16, 64, 512, 2048):
                x = rand_x(m, k)
                ref = (x.float() @ ref_w.T).to(torch.bfloat16)
                torch.cuda.synchronize()
                wrong = 0
                for _ in range(max(4, a.reps // 4)):
                    with torch.cuda.stream(side):
                        z = busy @ busy
                        xi = (x.float() + z[0, 0] * 0).to(torch.bfloat16)
                        got = run(xi).clone()
                    torch.cuda.synchronize()
                    wrong += int(not torch.equal(got, ref))
                stats["order"][0] += 1
                if wrong:
                    stats["order"][1] += 1
                    print(f"{name} N={n} K={k} M={m} side stream: {wrong} calls differ from the exact product",
                          flush=True)
            del layer, busy
    sw, rp, st = stats["sweep"], stats["repeat"], stats["streams"]
    print(f"MARLIN sweep: {sw[1]} of {sw[0]} (projection, M) calls differ from the exact product", flush=True)
    print(f"MARLIN repeats: {rp[0]} cases x {a.reps}; nondeterministic {rp[1]}; wrong {rp[2]}", flush=True)
    print(f"MARLIN two streams: {st[0]} cases x {a.reps}; wrong {st[1]}", flush=True)
    print(f"MARLIN side-stream ordering: {stats['order'][0]} cases; wrong {stats['order'][1]}", flush=True)


if __name__ == "__main__":
    main()
