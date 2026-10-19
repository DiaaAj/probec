"""What the probe detects (docs/REPORT.md §3.2): controlled faults in vLLM's real paged KV cache and in the softmax
scale, with detection, localization and latency per fault.

Part "kv" (CUDA graphs, prefix caching on): the prompts' blocks stay cached; one block (all its physical
copies) of one request is corrupted, the request reruns from the cache, and the output is compared with
the oracle. Structural faults are what KV bookkeeping bugs do: a block replaced by another block of the
same or another request, zeroed, holding the layer below's content, stale in one layer only, one slot or
one KV head replaced. Numeric faults are what kernel-numerics bugs do: K or V scaled by (1 + eps), with
Gaussian noise of relative size eps added, negated, or with the sign of a random fraction eps of its
entries flipped. Targets are blocks the oracle's walk reads; the
"random" variants pick any cached block, to show coverage (unread blocks cannot change the output).

Part "scale" (eager, so the change reaches every attention call): every attention layer's softmax scale
multiplied by a factor, on a twin with the scale check.

    python tools/fault_boundary.py kv <twin> --out out/boundary.jsonl
    python tools/fault_boundary.py scale <twin-with-scale-check> --out out/boundary.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def perturb(worker, blocks, kind, nkv, hd, src=None, layers=None, slot=None, src_slot=None, head=None, part=None,
            eps=0.0, seed=0):
    """Runs inside the vLLM worker: apply one fault to physical blocks. Returns the number of stored cache
    values that changed (0: the fault did not land, e.g. a perturbation that bf16 rounds away)."""
    import torch
    from probec.harness.localize import _kv_view
    mr = worker.model_runner
    nb = int(mr.kv_cache_config.num_blocks)
    bs = int(worker.cache_config.block_size)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    changed = 0
    for li, t in enumerate(mr.kv_caches):
        if layers is not None and li not in layers:
            continue
        for x in (t if isinstance(t, (list, tuple)) else [t]):
            v = _kv_view(x, nb, bs, nkv, hd)                       # (blocks, K/V, slots, kv heads, head dim)
            for i, b in enumerate(blocks):
                before = v[b].clone()
                if kind == "zero":
                    v[b].zero_()
                elif kind == "copy":
                    v[b].copy_(v[src[i]])
                elif kind == "slot":
                    v[b, :, slot].copy_(v[src[i], :, src_slot])
                elif kind == "head":
                    v[b, :, :, head].copy_(v[src[i], :, :, head])
                elif kind == "scale":
                    v[b, part].mul_(1.0 + eps)
                elif kind == "noise":
                    y = v[b, part]
                    r = y.float().pow(2).mean().sqrt()
                    y.add_((torch.randn(tuple(y.shape), generator=gen).to(y.device) * eps * r).to(y.dtype))
                elif kind == "negate":
                    v[b, part].neg_()
                elif kind == "flip":                                # sign of a random fraction eps of entries
                    y = v[b, part]
                    m = (torch.rand(tuple(y.shape), generator=gen) < eps).to(y.device)
                    y.mul_(torch.where(m, -1.0, 1.0).to(y.dtype))
                else:
                    raise ValueError(kind)
                changed += int((v[b] != before).sum())
    return changed


def set_scale(worker, factor):
    """Runs inside the vLLM worker: multiply every attention layer's softmax scale by `factor` (1 restores)."""
    n = 0
    for m in worker.model_runner.model.modules():
        impl = getattr(m, "impl", None)
        if impl is not None and isinstance(getattr(impl, "scale", None), float):
            if not hasattr(impl, "_probec_scale"):
                impl._probec_scale = impl.scale
            impl.scale = impl._probec_scale * factor
            n += 1
    return n


EPS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
FLIP = [0.01, 0.03, 0.1, 0.3]


def faults(L, trials, num_trials):
    """(name, params, trials): structural faults on read blocks, their random-block variants, numeric sweeps."""
    s = [("block_same_request", {"kind": "copy", "src": "same"}, trials),
         ("block_other_request", {"kind": "copy", "src": "other"}, trials),
         ("block_zeroed", {"kind": "zero"}, trials),
         ("block_layer_below", {"kind": "wrong_layer"}, trials),
         ("block_stale_one_layer", {"kind": "copy", "src": "other", "layers": [L // 2]}, trials),
         ("one_slot", {"kind": "slot", "src": "other"}, trials),
         ("one_slot_one_layer", {"kind": "slot", "src": "other", "layers": [L // 2]}, trials),
         ("one_kv_head", {"kind": "head", "src": "other"}, trials),        # kv head 0, the one the probe reads
         ("block_other_request_random", {"kind": "copy", "src": "other", "select": "random"}, 2 * trials)]
    for part, pn in ((0, "k"), (1, "v")):
        s.append((f"negate_{pn}", {"kind": "negate", "part": part}, num_trials))
        for e in EPS:
            s.append((f"scale_{pn}", {"kind": "scale", "part": part, "eps": e}, num_trials))
            s.append((f"noise_{pn}", {"kind": "noise", "part": part, "eps": e}, num_trials))
        for e in FLIP:
            s.append((f"flip_{pn}", {"kind": "flip", "part": part, "eps": e}, num_trials))
    return s


def kv_part(a, emit):
    from probec.harness import workload
    from probec.harness.localize import _copy_layer, _generate_cached, block_owners, kv_slot_tokens
    from probec.harness.vllm_engine import _generate, _llm
    from probec.oracle import Probe
    probe = Probe(a.twin)
    spec = probe.meta["spec"]
    nkv, hd, L = spec["kv_heads"], spec["head_dim"], spec["layers"]
    codes = probe.codes.astype(np.float32)
    mask0, sd = probe.meta["masks"][0], probe.meta["slow_dims"]
    rng = np.random.default_rng(a.seed)
    max_len = 4000
    prompts = [p for p in workload(probe, rng, 4 * a.prompts, max_len) if 600 <= len(p) <= max_len][:a.prompts]
    n = a.new_tokens
    exp = [probe.expected(p, n) for p in prompts]
    first_read = []                       # per request: prompt position -> first generation step that reads it
    for p, e in zip(prompts, exp):
        fr = {}
        for s in range(n):
            for pos in probe.reads(p, e, s):
                if pos < len(p):
                    fr.setdefault(int(pos), s)
        first_read.append(fr)
    t0 = time.time()
    llm = _llm(a.twin, max_len, n, {"attention_backend": "FLASH_ATTN", "gpu_memory_utilization": 0.85,
                                    "enable_prefix_caching": True})
    import vllm

    def remap():
        _generate(llm, prompts, 1)
        info = llm.collective_rpc(kv_slot_tokens, args=(codes, mask0, sd, nkv, hd))[0]
        bs = info["block_size"]
        return bs, block_owners(info["tokens"], prompts, bs), block_owners(info["tokens"], prompts, bs, all_copies=True)

    bs, owners, copies = remap()
    cov = []
    for r, p in enumerate(prompts):
        nblk = (len(p) - 1) // bs
        cov.append(len({pos // bs for pos in first_read[r] if pos // bs < nblk}) / max(1, nblk))
    emit({"part": "kv_header", "engine_version": vllm.__version__, "twin": os.path.basename(a.twin.rstrip("/")),
          "prompts": len(prompts), "prompt_lens": [len(p) for p in prompts], "new_tokens": n, "block_size": bs,
          "layers": L, "mapped_blocks": len(owners), "read_block_fraction": [round(c, 3) for c in cov],
          "init_sec": round(time.time() - t0, 1)})
    # baseline: every request exact from the cache before any fault
    base_bad = 0
    for r, p in enumerate(prompts[:4]):
        got, _ = _generate_cached(llm, p, n)
        base_bad += int(got != exp[r])
    emit({"part": "kv_baseline", "requests": 4, "wrong": base_bad})
    seed = 1000
    for name, f, trials in faults(L, a.trials, a.num_trials):
        for t in range(trials):
            seed += 1
            cached_blocks = {(r, j) for (r, j) in owners if j < (len(prompts[r]) - 1) // bs}
            if f.get("select") == "random":
                cands = sorted(cached_blocks)
            elif f["kind"] == "slot":
                cands = sorted({(r, pos // bs) for r in range(len(prompts)) for pos in first_read[r]} & cached_blocks)
            else:
                cands = sorted({(r, pos // bs) for r in range(len(prompts)) for pos in first_read[r]} & cached_blocks)
            if not cands:
                break
            r, j = cands[int(rng.integers(0, len(cands)))]
            b = owners[(r, j)]
            victims = list(copies.get((r, j), [b]))
            pool = [x for (rr, jj), x in owners.items() if (rr == r and jj != j) == (f.get("src") == "same")
                    and (rr != r or f.get("src") == "same") and x not in victims]
            src = [int(pool[int(rng.integers(0, len(pool)))])] * len(victims) if pool else None
            rec = {"part": "kv", "fault": name, "eps": f.get("eps"), "request": r, "block": j, "trial": t}
            slot_pos = None
            if f["kind"] == "wrong_layer":
                for vb in victims:
                    for l in range(L - 1, 0, -1):
                        llm.collective_rpc(_copy_layer, args=(vb, l - 1, l, nkv, hd))
                changed = -1
            elif f["kind"] in ("copy", "slot", "head") and src is None:
                continue
            else:
                kw = {"layers": f.get("layers"), "part": f.get("part"), "eps": f.get("eps", 0.0), "seed": seed}
                if f["kind"] == "slot":
                    in_block = [pos for pos in first_read[r] if pos // bs == j]
                    slot_pos = in_block[int(rng.integers(0, len(in_block)))]
                    kw.update(slot=slot_pos % bs, src_slot=int(rng.integers(0, bs)))
                if f["kind"] == "head":
                    kw.update(head=0)                               # other kv heads carry no information in a twin
                changed = llm.collective_rpc(perturb, args=(victims, f["kind"], nkv, hd, src), kwargs=kw)[0]
            got, cached = _generate_cached(llm, prompts[r], n)
            diffs = [i for i, (g, e) in enumerate(zip(got, exp[r])) if g != e]
            if slot_pos is not None:
                read_step = first_read[r].get(slot_pos)
            else:
                steps = [s for pos, s in first_read[r].items() if pos // bs == j]
                read_step = min(steps) if steps else None
            rec.update(slot_pos=slot_pos, layers=f.get("layers"), src_blocks=src[:1] if src else None,
                       changed=changed, cache_hit_tokens=cached, read_step=read_step, detected=bool(diffs),
                       first_bad_step=diffs[0] if diffs else None, n_bad=len(diffs))
            if diffs:
                reads = [pos for pos in probe.reads(prompts[r], exp[r], diffs[0]) if pos < len(prompts[r])]
                rec["localized"] = (slot_pos in reads) if slot_pos is not None else (j in {pos // bs for pos in reads})
            emit(rec)
            llm.reset_prefix_cache()                                   # drop the corrupted block
            bs, owners, copies = remap()


def scale_part(a, emit):
    from probec.harness import score, workload
    from probec.harness.vllm_engine import _generate, _llm
    from probec.oracle import Probe
    probe = Probe(a.twin)
    rng = np.random.default_rng(a.seed)
    max_len = 8000
    prompts = workload(probe, rng, a.scale_requests, max_len)
    llm = _llm(a.twin, max_len, a.new_tokens, {"attention_backend": "FLASH_ATTN", "gpu_memory_utilization": 0.85,
                                               "enforce_eager": True, "enable_prefix_caching": False})
    import vllm
    factors = [1.0, 2.0, 0.5, math.sqrt(2), 1 / math.sqrt(2), 1 / math.log(2), math.log(2), 1.5, 1 / 1.5,
               1.4, 1 / 1.4, 1.35, 1 / 1.35, 1.3, 1 / 1.3, 1.25, 0.8, 1.1, 0.9, 1.03, 0.97, 1.01, 0.99]
    for fct in factors:
        layers = llm.collective_rpc(set_scale, args=(fct,))[0]
        diag = {}
        bt, br, fb = score(probe, prompts, _generate(llm, prompts, a.new_tokens), a.new_tokens, diag)
        emit({"part": "scale", "engine_version": vllm.__version__, "factor": round(fct, 6), "layers_patched": layers,
              "requests": len(prompts), "wrong_requests": br, "wrong_tokens": bt,
              "scale_diagnosed": diag.get("softmax scale", 0),
              "median_first_bad_step": float(np.median(fb)) if fb else None})
    llm.collective_rpc(set_scale, args=(1.0,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["kv", "scale"])
    ap.add_argument("twin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=12, help="trials per structural fault")
    ap.add_argument("--num-trials", type=int, default=6, help="trials per numeric fault and size")
    ap.add_argument("--scale-requests", type=int, default=48)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def emit(rec):
        print("BOUNDARY", json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    (kv_part if a.part == "kv" else scale_part)(a, emit)


if __name__ == "__main__":
    main()
