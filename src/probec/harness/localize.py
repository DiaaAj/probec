"""Targeted KV-cache fault injection with localization, using the probe's self-describing KV cache.

Every token's key in layer 0 is SK * (code(token) ⊙ mask_0) in the slow RoPE dims of kv head 0, so the
sign pattern of a cached key names the token stored in that slot. Decoding the whole cache maps every
physical block to the tokens it holds, which identifies the blocks of a given request without touching
engine internals beyond the raw cache tensors."""
import numpy as np


def _layout(x, nb, bs, nkv, hd):
    """Permutation that views a per-layer cache tensor as (num_blocks, 2[K,V], block_size, kv_heads, hd)."""
    shape = list(x.shape)
    want = {"nb": nb, "kv": 2, "bs": bs, "h": nkv, "d": hd}
    dims = {}
    for key in ("nb", "d", "bs", "h", "kv"):
        cands = [i for i, s in enumerate(shape) if s == want[key] and i not in dims.values()]
        if not cands:
            raise RuntimeError(f"cannot find dim {key}={want[key]} in cache shape {shape}")
        dims[key] = cands[0]
    return [dims["nb"], dims["kv"], dims["bs"], dims["h"], dims["d"]]


def _kv_view(x, nb, bs, nkv, hd):
    """View of a per-layer cache tensor as (num_blocks, 2, block_size, kv_heads, hd). Handles layouts that
    fuse K and V in the last dim (e.g. vLLM 0.31 FLASH_ATTN: [nb, kv_heads, bs, 2*hd]); the self-describing
    decode verifies the K half (a wrong guess decodes no tokens)."""
    shape = list(x.shape)
    if hd not in shape and 2 * hd in shape:
        x = x.unflatten(shape.index(2 * hd), (2, hd))
    return x.permute(*_layout(x, nb, bs, nkv, hd))


def kv_slot_tokens(worker, codes, mask0, slow_dims, nkv, hd, layer=0):
    """Runs inside a vLLM worker: token id held by every (block, slot) of `layer`, or -1."""
    import torch
    mr = worker.model_runner
    caches = mr.kv_caches
    x = caches[layer]
    x = x[0] if isinstance(x, (list, tuple)) else x
    nb = int(mr.kv_cache_config.num_blocks)
    bs = int(worker.cache_config.block_size)
    k = _kv_view(x, nb, bs, nkv, hd)[:, 0, :, 0, :]                 # (nb, bs, hd), kv head 0
    sig = torch.sign(k[..., slow_dims].float()) * torch.tensor(mask0, device=k.device, dtype=torch.float32)
    book = torch.tensor(np.asarray(codes, np.float32), device=k.device)   # (n_codes, code)
    flat = sig.reshape(-1, sig.shape[-1])
    out = torch.full((flat.shape[0],), -1, dtype=torch.int64, device=k.device)
    for s in range(0, flat.shape[0], 65536):
        sc = flat[s:s + 65536] @ book.T
        best, idx = sc.max(dim=1)
        out[s:s + 65536] = torch.where(best == book.shape[1], idx, torch.full_like(idx, -1))
    return {"block_size": bs, "num_blocks": nb, "tokens": out.reshape(nb, bs).cpu().numpy().tolist()}


def corrupt(worker, blocks, mode, src_blocks=None, layers=None, nkv=None, hd=None):
    """Runs inside a vLLM worker: overwrite physical blocks (all layers, or only `layers`)."""
    if mode != "zero" and src_blocks is None:
        raise ValueError("copy mode needs src_blocks")
    src = list(src_blocks or [])
    mr = worker.model_runner
    nb = int(mr.kv_cache_config.num_blocks)
    bs = int(worker.cache_config.block_size)
    for li, t in enumerate(mr.kv_caches):
        if layers is not None and li not in layers:
            continue
        for x in (t if isinstance(t, (list, tuple)) else [t]):
            v = _kv_view(x, nb, bs, nkv, hd)
            for i, b in enumerate(blocks):
                if mode == "zero":
                    v[b].zero_()
                else:
                    v[b].copy_(v[src[i]])
    return True


def block_owners(slot_tokens, prompts, block_size, all_copies=False):
    """Map (request, prompt block index) -> physical block id, by matching full-block token sequences.
    Concurrent prefills of shared prefixes can leave several physical copies of the same content; with
    all_copies=True the value is the list of every copy (corrupt them all, or a rerun may read another)."""
    by_content = {}
    for b, toks in enumerate(slot_tokens):
        if toks[0] >= 0:
            by_content.setdefault(tuple(toks), []).append(b)
    owners = {}
    for r, p in enumerate(prompts):
        for j in range(len(p) // block_size):
            chunk = tuple(p[j * block_size:(j + 1) * block_size])
            if chunk in by_content:
                owners[(r, j)] = by_content[chunk] if all_copies else by_content[chunk][0]
    return owners


def targeted_campaign(llm, probe, prompts, n_new, rng, generate, trials=8):
    """For each fault kind: corrupt one block of one request's cached prompt, rerun, check detection
    and that the first wrong token's reads include the corrupted block."""
    meta = probe.meta
    spec = meta["spec"]
    codes = probe.codes.astype(np.float32)
    mask0 = meta["masks"][0]
    sd = meta["slow_dims"]
    nkv, hd = spec["kv_heads"], spec["head_dim"]
    generate(llm, prompts, 1)                                   # make every prompt's blocks resident
    info = llm.collective_rpc(kv_slot_tokens, args=(codes, mask0, sd, nkv, hd))[0]
    bs = info["block_size"]
    owners = block_owners(info["tokens"], prompts, bs)
    copies = block_owners(info["tokens"], prompts, bs, all_copies=True)
    res = {}
    for kind in ("block_dup", "zero_block", "cross_request", "wrong_layer", "stale_partial"):
        r_stats = {"trials": 0, "detected": 0, "localized": 0, "first_bad_steps": []}
        cands = [(r, j) for (r, j) in owners
                 if any(t < probe.C for t in prompts[r][j * bs:(j + 1) * bs])]
        for _ in range(min(trials, len(cands))):
            r, j = cands[int(rng.integers(0, len(cands)))]
            b = owners[(r, j)]
            if kind == "block_dup":            # block table points at another block of the same request
                others = [x for (rr, jj), x in owners.items() if rr == r and jj != j and x != b]
            else:                              # a block of a different request
                others = [x for (rr, jj), x in owners.items() if rr != r and x != b]
            if not others:
                continue
            src = [int(others[int(rng.integers(0, len(others)))])]
            L = spec["layers"]
            victims = list(copies.get((r, j), [b]))
            if kind == "zero_block":
                args = (victims, "zero", None, None, nkv, hd)
            elif kind == "stale_partial":
                args = (victims, "copy", src * len(victims), [L // 2], nkv, hd)
            elif kind == "wrong_layer":
                args = None
            else:
                args = (victims, "copy", src * len(victims), None, nkv, hd)
            if kind == "wrong_layer":
                # every layer l reads layer l-1's content for this block: copy layer by layer, top down
                for vb in victims:
                    for l in range(L - 1, 0, -1):
                        llm.collective_rpc(_copy_layer, args=(vb, l - 1, l, nkv, hd))
            else:
                llm.collective_rpc(corrupt, args=args)
            # diagnostics: did the corruption land (re-decode the block), and does the rerun hit the cache?
            after = llm.collective_rpc(kv_slot_tokens, args=(codes, mask0, sd, nkv, hd))[0]["tokens"][b]
            landed = after != info["tokens"][b]
            got, cached = _generate_cached(llm, prompts[r], n_new)
            r_stats.setdefault("landed", 0)
            r_stats.setdefault("cache_hit_tokens", [])
            r_stats["landed"] += int(landed)
            r_stats["cache_hit_tokens"].append(cached)
            exp = probe.expected(prompts[r], n_new)
            diffs = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
            r_stats["trials"] += 1
            if diffs:
                r_stats["detected"] += 1
                r_stats["first_bad_steps"].append(diffs[0])
                read_blocks = {pos // bs for pos in probe.reads(prompts[r], exp, diffs[0])}
                r_stats["localized"] += int(j in read_blocks)
            llm.reset_prefix_cache()                            # drop the corrupted block
            generate(llm, prompts, 1)
            info = llm.collective_rpc(kv_slot_tokens, args=(codes, mask0, sd, nkv, hd))[0]
            owners = block_owners(info["tokens"], prompts, bs)
            copies = block_owners(info["tokens"], prompts, bs, all_copies=True)
            cands = [(rr, jj) for (rr, jj) in owners
                     if any(t < probe.C for t in prompts[rr][jj * bs:(jj + 1) * bs])]
        s = r_stats.pop("first_bad_steps")
        r_stats["median_first_bad_step"] = float(np.median(s)) if s else None
        ch = r_stats.pop("cache_hit_tokens", [])
        r_stats["median_cache_hit_tokens"] = float(np.median(ch)) if ch else None
        res[kind] = r_stats
    return {"mapped_blocks": len(owners), "kinds": res}


def _generate_cached(llm, prompt, n_new):
    """Greedy generation for one prompt; returns (tokens, number of prompt tokens served from the cache)."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_new, ignore_eos=True, detokenize=False, temperature=0.0)
    o = llm.generate([{"prompt_token_ids": prompt}], sp, use_tqdm=False)[0]
    return list(o.outputs[0].token_ids), int(getattr(o, "num_cached_tokens", -1) or -1)


def _copy_layer(worker, block, src_layer, dst_layer, nkv, hd):
    mr = worker.model_runner
    nb = int(mr.kv_cache_config.num_blocks)
    bs = int(worker.cache_config.block_size)
    xs = mr.kv_caches[src_layer]
    xd = mr.kv_caches[dst_layer]
    xs = xs[0] if isinstance(xs, (list, tuple)) else xs
    xd = xd[0] if isinstance(xd, (list, tuple)) else xd
    vs = _kv_view(xs, nb, bs, nkv, hd)
    vd = _kv_view(xd, nb, bs, nkv, hd)
    vd[block].copy_(vs[block])
    return True
