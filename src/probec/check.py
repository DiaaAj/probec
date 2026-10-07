"""CPU checks with HF transformers: zero false alarms, bit-exact internal states, and KV fault injection
with detection and localization. Slow but engine-independent; the reference semantics for every probe."""
import json
import time

import numpy as np
import torch

from .oracle import Probe

BLK = 16


def load(model_dir, attn, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
    return m.eval()


@torch.no_grad()
def batched_generate(model, probe, prompts, n_new, sample):
    """Left-padded batch generation; returns (mismatching tokens, total tokens)."""
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), probe.pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):] = torch.tensor(p)
        mask[i, width - len(p):] = 1
    kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_new, min_new_tokens=n_new,
                         pad_token_id=probe.pad, **kw)[:, width:]
    bad = 0
    for i, p in enumerate(prompts):
        exp = probe.expected(p, n_new)
        bad += sum(int(a != b) for a, b in zip(out[i].tolist(), exp))
    return bad, len(prompts) * n_new


@torch.no_grad()
def bitwise_internal(model, probe, prompt, n_new):
    """Teacher-force prompt + expected continuation; every layer's residual vector at every checked
    position must equal the canonical vector bit for bit."""
    exp = probe.expected(prompt, n_new)
    seq = prompt + exp[:-1]
    out = model(input_ids=torch.tensor([seq]), output_hidden_states=True)
    hs = out.hidden_states                  # embeddings + one per layer (the last is post-norm)
    emb = model.get_input_embeddings().weight
    first = len(prompt) - 1
    mism = checked = 0
    for pos in range(first, len(seq)):
        base = emb[seq[pos]].float()
        for l, fields in probe.field_states(seq, pos).items():
            canon = base.clone()
            for off, vals in fields.items():
                canon[off:off + len(vals)] = torch.from_numpy(vals)
            checked += 1
            mism += int(not torch.equal(hs[l + 1][0, pos].float(), canon))
    logits_ok = out.logits[0, first:].argmax(-1).tolist() == exp
    return checked, mism, logits_ok


@torch.no_grad()
def prefill(model, ids):
    from transformers import DynamicCache
    try:
        cache = DynamicCache(config=model.config)       # builds indexed layers for DSA models
    except TypeError:
        cache = DynamicCache()
    out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
    return out.logits[0, -1].argmax().item(), cache


@torch.no_grad()
def decode(model, cache, first_tok, n, start_pos):
    toks, tok = [first_tok], first_tok
    for i in range(n - 1):
        mask = torch.ones((1, start_pos + i + 1), dtype=torch.long)
        out = model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True,
                    attention_mask=mask, position_ids=torch.tensor([[start_pos + i]]))
        tok = out.logits[0, -1].argmax().item()
        toks.append(tok)
    return toks


def clone_cache(cache):
    import copy
    return copy.deepcopy(cache)


FAULTS = ["block_dup", "cross_request", "zero_block", "wrong_layer", "stale_partial"]


@torch.no_grad()
def fault_campaign(model, probe, rng, n_trials, n_new, length):
    res = {k: {"trials": 0, "detected": 0, "steps_to_detect": [], "localized": 0} for k in FAULTS}
    for _ in range(n_trials):
        pool = int(rng.integers(0, len(probe.pools)))
        p = probe.prompt(pool, length, rng)
        q = probe.prompt((pool + 1) % len(probe.pools), length, rng)
        exp = probe.expected(p, n_new)
        tok0, cache = prefill(model, p)
        if tok0 != exp[0]:
            raise AssertionError("clean prefill mismatch")
        _, cache_q = prefill(model, q)
        nblk = len(p) // BLK
        # victim blocks are read during the corrupted decode (steps >= 1; a block nobody reads cannot
        # change the output); fall back to blocks holding a class token
        read = {pos // BLK for s in range(1, n_new) for pos in probe.reads(p, exp, s)}
        cand = sorted(b for b in read if b < nblk) or \
            [b for b in range(nblk) if any(x < probe.C for x in p[b * BLK:(b + 1) * BLK])]
        for kind in FAULTS:
            b = int(rng.choice(cand))
            sl = slice(b * BLK, (b + 1) * BLK)
            c = clone_cache(cache)
            for l, layer in enumerate(c.layers):
                if kind == "block_dup":            # block table points at another block of the request
                    src = int(rng.choice([x for x in range(nblk) if x != b]))
                    layer.keys[:, :, sl] = cache.layers[l].keys[:, :, src * BLK:(src + 1) * BLK]
                    layer.values[:, :, sl] = cache.layers[l].values[:, :, src * BLK:(src + 1) * BLK]
                elif kind == "cross_request":      # block from a concurrent request (another pool)
                    layer.keys[:, :, sl] = cache_q.layers[l].keys[:, :, sl]
                    layer.values[:, :, sl] = cache_q.layers[l].values[:, :, sl]
                elif kind == "zero_block":         # freshly allocated, never-written block
                    layer.keys[:, :, sl] = 0
                    layer.values[:, :, sl] = 0
                elif kind == "wrong_layer" and l > 0:   # layer-offset bug: read the previous layer's KV
                    layer.keys[:, :, sl] = cache.layers[l - 1].keys[:, :, sl]
                    layer.values[:, :, sl] = cache.layers[l - 1].values[:, :, sl]
                elif kind == "stale_partial" and l == len(c.layers) // 2:   # one layer, one block
                    layer.keys[:, :, sl] = cache_q.layers[l].keys[:, :, sl]
                    layer.values[:, :, sl] = cache_q.layers[l].values[:, :, sl]
            got = decode(model, c, exp[0], n_new, len(p))
            r = res[kind]
            r["trials"] += 1
            diffs = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
            if diffs:
                r["detected"] += 1
                r["steps_to_detect"].append(diffs[0])
                blocks_read = {pos // BLK for pos in probe.reads(p, exp, diffs[0])}
                r["localized"] += int(b in blocks_read)
    for r in res.values():
        s = r.pop("steps_to_detect")
        r["median_steps_to_detect"] = float(np.median(s)) if s else None
    return res


@torch.no_grad()
def decode_positions(model, cache, first_tok, n, positions):
    """Greedy decode where step i uses position positions[i] (to simulate position-id bugs)."""
    toks, tok = [first_tok], first_tok
    seen = cache.get_seq_length()
    for i in range(n - 1):
        mask = torch.ones((1, seen + i + 1), dtype=torch.long)
        out = model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True,
                    attention_mask=mask, position_ids=torch.tensor([[positions[i]]]))
        tok = out.logits[0, -1].argmax().item()
        toks.append(tok)
    return toks


POS_FAULTS = ["decode_position_shift", "key_position_skew", "rope_theta_drift"]


@torch.no_grad()
def position_faults(model, probe, rng, n_trials, n_new, length):
    """Position/RoPE faults for the positional probe: a +1 position-id slip during decode, one block of
    cached keys rotated as if stored one position late, and RoPE frequencies off by 5%."""
    res = {k: {"trials": 0, "detected": 0, "steps_to_detect": [], "localized": 0} for k in POS_FAULTS}
    rot = model.model.rotary_emb
    inv = rot.inv_freq.clone()
    half = inv.numel()
    for _ in range(n_trials):
        p = probe.sample_prompt(rng, length)
        exp = probe.expected(p, n_new)
        for kind in POS_FAULTS:
            victim = None
            if kind == "rope_theta_drift":
                rot.inv_freq.copy_(inv * 1.05)
            tok0, cache = prefill(model, p)
            if kind == "rope_theta_drift":
                got = [tok0] + decode(model, cache, tok0, n_new, len(p))[1:]
                rot.inv_freq.copy_(inv)
            elif kind == "decode_position_shift":     # positions skip one slot from the 3rd decode step
                pos = [len(p) + i + (1 if i >= 2 else 0) for i in range(n_new)]
                got = decode_positions(model, cache, tok0, n_new, pos)
            else:                                      # keys of one block carry position+1 rotation
                # victim: a full prompt block read by a decode step (step 0 runs in the clean prefill)
                body0 = probe.min_len - 2                     # the identical-token prefix ends here
                read = sorted({q // BLK for st in range(1, n_new) for q in probe.reads(p, exp, st)
                               if body0 <= q < (len(p) // BLK) * BLK and q // BLK * BLK >= body0})
                victim = int(read[int(rng.integers(0, len(read)))])
                sl = slice(victim * BLK, (victim + 1) * BLK)
                cos, sin = torch.cos(inv), torch.sin(inv)
                for layer in cache.layers:
                    k = layer.keys[:, :, sl].float()
                    k1, k2 = k[..., :half], k[..., half:]
                    layer.keys[:, :, sl] = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], -1).to(layer.keys.dtype)
                got = decode(model, cache, exp[0], n_new, len(p))
            r = res[kind]
            r["trials"] += 1
            diffs = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
            if diffs:
                r["detected"] += 1
                r["steps_to_detect"].append(diffs[0])
                if victim is not None:
                    blocks_read = {q // BLK for q in probe.reads(p, exp, diffs[0])}
                    r["localized"] += int(victim in blocks_read)
    for r in res.values():
        st = r.pop("steps_to_detect")
        r["median_steps_to_detect"] = float(np.median(st)) if st else None
    return res


ROUTE_FAULTS = ["misroute", "drop", "duplicate"]


@torch.no_grad()
def routing_faults(model, probe, rng, n_trials, n_new, length):
    """MoE routing faults injected into HF's router at one layer and one decode step: send one routed
    slot to an unrouted expert, drop a slot, or duplicate one slot's expert into another. Detection must
    happen, and the first wrong token should be a diagnostic token naming an expert involved."""
    res = {k: {"trials": 0, "detected": 0, "named": 0, "steps_to_detect": []} for k in ROUTE_FAULTS}
    layers = probe.meta["moe_layers"][:-1]              # the last MoE layer's routing is unchecked
    E = probe.E
    for _ in range(n_trials):
        p = probe.sample_prompt(rng, length)
        exp = probe.expected(p, n_new)
        for kind in ROUTE_FAULTS:
            l = int(layers[int(rng.integers(0, len(layers)))])
            step = int(rng.integers(1, n_new - 1))
            router = model.model.layers[l].mlp.gate
            orig = router.forward
            calls = {"n": 0}
            involved = set()

            def patched(hidden_states, _orig=orig, _kind=kind, _step=step):
                logits, scores, idx = _orig(hidden_states)
                calls["n"] += 1
                if calls["n"] == _step + 1:                  # call 0 is the prefill
                    idx = idx.clone()
                    row = idx[-1]
                    sel = row.tolist()
                    a, b = 0, 1
                    if _kind == "misroute":
                        other = [e for e in range(E) if e not in sel]
                        new = other[int(rng.integers(0, len(other)))]
                        involved.update({sel[a], new})
                        row[a] = new
                    elif _kind == "drop":
                        involved.add(sel[a])
                        row[a] = E                            # HF skips index num_experts
                    else:
                        involved.update({sel[a], sel[b]})
                        row[b] = sel[a]
                return logits, scores, idx

            router.forward = patched
            try:
                tok0, cache = prefill(model, p)
                got = [tok0] + decode(model, cache, tok0, n_new, len(p))[1:]
            finally:
                router.forward = orig
            r = res[kind]
            r["trials"] += 1
            diffs = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
            if diffs:
                r["detected"] += 1
                r["steps_to_detect"].append(diffs[0])
                r["named"] += int(probe.diagnose(got[diffs[0]]) in involved)
    for r in res.values():
        st = r.pop("steps_to_detect")
        r["median_steps_to_detect"] = float(np.median(st)) if st else None
    return res


@torch.no_grad()
def dsa_query_drop(model, probe, rng, n_trials, n_new, length):
    """DSA fault (the vLLM #58704 class): during decode, the sparse-attention top-k set loses the query
    token itself. The echo token's layer-0 self-read then fails."""
    res = {"trials": 0, "detected": 0, "steps_to_detect": []}
    layers = [l for l in model.model.layers if hasattr(l.self_attn, "indexer")]
    origs = [l.self_attn.indexer.forward for l in layers]

    def make(orig):
        def patched(hidden_states, *args, **kw):
            idx = orig(hidden_states, *args, **kw)
            if hidden_states.shape[1] == 1:                  # decode step: drop the newest position
                last = idx.max(dim=-1, keepdim=True).values
                idx = torch.where(idx == last, idx.min(dim=-1, keepdim=True).values, idx)
            return idx
        return patched

    for _ in range(n_trials):
        p = probe.sample_prompt(rng, length)
        exp = probe.expected(p, n_new)
        for l, o in zip(layers, origs):
            l.self_attn.indexer.forward = make(o)
        try:
            tok0, cache = prefill(model, p)
            got = [tok0] + decode(model, cache, tok0, n_new, len(p))[1:]
        finally:
            for l, o in zip(layers, origs):
                l.self_attn.indexer.forward = o
        res["trials"] += 1
        diffs = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
        if diffs:
            res["detected"] += 1
            res["steps_to_detect"].append(diffs[0])
    st = res.pop("steps_to_detect")
    res["median_steps_to_detect"] = float(np.median(st)) if st else None
    return res


def run_checks(model_dir, seed=1, quick=False, n_new=24, faults=10, fault_len=600, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    probe = Probe(model_dir)
    report = {"model": model_dir, "torch": torch.__version__, "quick": quick}
    if quick:
        n_new, faults, fault_len = 12, 2, 300
        grid = {torch.float32: [(1, [200], False), (3, [150, 64, 260], True)],
                torch.bfloat16: [(2, [130, 200], False)]}
        attns = ["eager", "sdpa"]
    else:
        # bf16 matmuls are slow on CPUs without native bf16, so bf16 runs use shorter prompts
        grid = {torch.bfloat16: [(1, [300], False), (4, [150, 260, 64, 400], False), (3, [200, 129, 330], True)],
                torch.float32: [(1, [300], False), (4, [150, 700, 64, 1200], False), (4, [200, 900, 450, 129], True)]}
        attns = ["eager", "sdpa"]
    zfa = []
    for attn in attns:
        for dtype, runs in grid.items():
            model = load(model_dir, attn, dtype)
            for batch, lengths, sample in runs:
                prompts = [probe.sample_prompt(rng, n) for n in lengths]
                t0 = time.time()
                bad, tot = batched_generate(model, probe, prompts, n_new, sample)
                zfa.append({"attn": attn, "dtype": str(dtype), "batch": batch, "lengths": lengths,
                            "sample": sample, "tokens": tot, "mismatches": bad,
                            "sec": round(time.time() - t0, 1)})
                log(zfa[-1])
    report["zero_false_alarm"] = zfa
    model = load(model_dir, "sdpa", torch.bfloat16)
    bw = []
    for n in ([130] if quick else [130, 300]):
        prm = probe.sample_prompt(rng, n)
        checked, mism, ok = bitwise_internal(model, probe, prm, n_new)
        bw.append({"len": n, "layer_states_checked": checked, "non_bitwise": mism, "logits_ok": ok})
        log(bw[-1])
    report["bitwise_internal"] = bw
    model = load(model_dir, "sdpa", torch.float32)       # fault semantics are dtype-independent
    if probe.program == "posrecall":
        report["faults"] = position_faults(model, probe, rng, faults, n_new, max(fault_len, probe.min_len + 200))
    else:
        report["faults"] = fault_campaign(model, probe, rng, faults, n_new, fault_len)
    if probe.program == "moe":
        report["routing_faults"] = routing_faults(model, probe, rng, faults, n_new, fault_len)
        log(json.dumps(report["routing_faults"]))
    log(json.dumps(report["faults"]))
    report["ok"] = (sum(z["mismatches"] for z in zfa) == 0
                    and all(b["non_bitwise"] == 0 and b["logits_ok"] for b in bw))
    return report
