"""vLLM adapter: offline LLM engine with token-id prompts (no tokenizer), plus KV fault injection through
the real paged cache via collective_rpc."""
import gc
import os
import time

import numpy as np

from . import make_workload, score

BACKENDS = {"dense": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"],
            "mla": ["TRITON_MLA"]}
HOPPER = {"dense": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"], "mla": ["FLASHMLA", "FLASH_ATTN_MLA", "TRITON_MLA"]}
BLACKWELL = {"dense": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"], "mla": ["CUTLASS_MLA", "FLASHINFER_MLA", "TRITON_MLA"]}


def matrix(name, kind="dense"):
    """Serving configurations. kind: dense (GQA attention) or mla."""
    plan = []
    if name == "hopper":
        backends = HOPPER[kind]
    elif name == "blackwell":
        backends = BLACKWELL[kind]
    elif name in ("quick", "medium"):
        backends = BACKENDS[kind][:2]
    else:
        backends = BACKENDS[kind]
    for be in backends:
        base = {"attention_backend": be, "gpu_memory_utilization": 0.85}
        plan.append({"name": f"{be}/default", "kw": dict(base, enable_prefix_caching=True),
                     "faults": be == backends[0] and name not in ("quick",) and kind == "dense"})
        if name == "quick":
            continue
        plan.append({"name": f"{be}/chunk512_sample", "sample": True,
                     "kw": dict(base, enable_prefix_caching=True, max_num_batched_tokens=512)})
        plan.append({"name": f"{be}/eager_preempt",
                     "kw": dict(base, enforce_eager=True, enable_prefix_caching=False,
                                num_gpu_blocks_override=2500)})
    if name not in ("quick",):
        fp8 = {"dense": ["FLASHINFER", "TRITON_ATTN"], "mla": backends[:1]}[kind]
        if name in ("hopper", "blackwell") and kind == "dense":
            fp8 = ["FLASH_ATTN"] + fp8                 # FA3 FP8 also quantizes the query
        for be in fp8:
            plan.append({"name": f"{be}/fp8kv", "kw": {"attention_backend": be, "kv_cache_dtype": "fp8",
                                                       "gpu_memory_utilization": 0.85}})
        plan.append({"name": f"{backends[0]}/ngram_spec",
                     "kw": {"attention_backend": backends[0], "gpu_memory_utilization": 0.85,
                            "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                                   "prompt_lookup_max": 4, "prompt_lookup_min": 2}}})
        plan.append({"name": f"{backends[0]}/ngram_spec_decoys", "workload": "decoys",
                     "kw": {"attention_backend": backends[0], "gpu_memory_utilization": 0.85,
                            "enable_prefix_caching": True,
                            "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                                   "prompt_lookup_max": 4, "prompt_lookup_min": 2}}})
    if name == "replay":       # version-neutral (no backend kwargs): for replaying old releases
        plan = [{"name": "default/prefix", "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True}},
                {"name": "default/shared_prefix", "workload": "shared",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True}},
                {"name": "default/shared_prefix_long_suffix", "workload": "shared_long",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True}},
                {"name": "default/split_markers", "workload": "split_markers",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True}},
                # MLA prefill backends that merge cached-context and suffix attention by log-sum-exp
                {"name": "default/split_markers_trtllm_ragged", "workload": "split_markers",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True,
                        "attention_config": {"mla_prefill_backend": "TRTLLM_RAGGED"}}},
                {"name": "default/split_markers_flashinfer_prefill", "workload": "split_markers",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True,
                        "attention_config": {"mla_prefill_backend": "FLASHINFER"}}},
                {"name": "default/fp8kv", "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True,
                                                  "kv_cache_dtype": "fp8"}},
                {"name": "default/ngram_spec",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True,
                        "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2}}},
                {"name": "default/ngram_spec_async",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True, "async_scheduling": True,
                        "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2}}},
                {"name": "default/ngram_spec_decoys", "workload": "decoys",
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True,
                        "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2}}},
                {"name": "default/ngram_gpu_async_decoys", "workload": "decoys",   # async needs the GPU drafter
                 "kw": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True, "async_scheduling": True,
                        "speculative_config": {"method": "ngram_gpu", "num_speculative_tokens": 4,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2}}}]
    if name == "mladebug":     # isolate MLA prefill-path failures (version-neutral kwargs)
        base = {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True}
        plan = [{"name": "mla/default", "seed": 0, "kw": dict(base)},
                {"name": "mla/no_prefix_cache", "seed": 0, "kw": dict(base, enable_prefix_caching=False)},
                {"name": "mla/eager", "seed": 0, "kw": dict(base, enforce_eager=True)},
                {"name": "mla/big_batch_budget", "seed": 0, "kw": dict(base, max_num_batched_tokens=65536)},
                {"name": "mla/small_batch_budget", "seed": 0, "kw": dict(base, max_num_batched_tokens=1024)},
                {"name": "mla/no_chunked_prefill", "seed": 0, "kw": dict(base, enable_chunked_prefill=False,
                                                                         max_num_batched_tokens=8192)},
                # cached context per step beyond the MLA prefill workspace: forces multi-chunk context
                {"name": "mla/shared_long_big_budget", "seed": 0, "workload": "shared_long",
                 "kw": dict(base, max_num_batched_tokens=65536)},
                {"name": "mla/big_budget_many_seqs", "seed": 0,
                 "kw": dict(base, max_num_batched_tokens=65536, max_num_seqs=256)}]
    if name == "ep2":
        plan = [{"name": f"{be}/ep2_tp2", "kw": {"attention_backend": be, "gpu_memory_utilization": 0.85,
                                                  "tensor_parallel_size": 2, "enable_expert_parallel": True,
                                                  "enable_prefix_caching": True}} for be in backends[:1]]
        plan.append({"name": f"{backends[0]}/tp2", "kw": {"attention_backend": backends[0],
                                                          "gpu_memory_utilization": 0.85, "tensor_parallel_size": 2}})
        plan.append({"name": f"{backends[0]}/ep2_eplb", "kw": {"attention_backend": backends[0],
                                                               "gpu_memory_utilization": 0.85, "tensor_parallel_size": 2,
                                                               "enable_expert_parallel": True, "enable_eplb": True}})
    return plan


def _llm(model, max_len, n_new, kw):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    stats = {"disable_log_stats": False} if "speculative_config" in kw else {}   # spec-decode counters
    return LLM(model=model, skip_tokenizer_init=True, dtype="bfloat16", seed=0,
               max_model_len=max_len + n_new + 128, **stats, **kw)


def spec_stats(llm):
    """Speculative-decoding counters (drafts, draft tokens, accepted tokens) from vLLM's metrics."""
    try:
        out = {}
        for m in llm.get_metrics():
            if "spec_decode" not in m.name:
                continue
            v = getattr(m, "value", None)
            if v is None and getattr(m, "values", None) is not None:
                v = sum(m.values)
            if isinstance(v, (int, float)):
                k = m.name.split("spec_decode_")[-1]
                out[k] = out.get(k, 0) + v
        return out
    except Exception as e:
        return {"error": repr(e)[:200]}


def _generate(llm, prompts, n_new, temperature=0.0):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_new, ignore_eos=True, detokenize=False, temperature=temperature)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs]


def corrupt_blocks(worker, n_blocks, seed, mode):
    """Runs inside the vLLM worker: overwrite n random physical KV blocks in every layer."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    mr = worker.model_runner
    caches = mr.kv_caches
    first = caches[0][0] if isinstance(caches[0], (list, tuple)) else caches[0]
    cfg = getattr(mr, "kv_cache_config", None)
    nb = int(cfg.num_blocks) if cfg is not None else max(first.shape)
    dst = torch.randint(1, nb, (n_blocks,), generator=g)
    src = torch.randint(1, nb, (n_blocks,), generator=g)
    for t in caches:
        for x in (t if isinstance(t, (list, tuple)) else [t]):
            dim = [i for i, s in enumerate(x.shape) if s == nb][0]
            idx = dst.to(x.device)
            if mode == "zero":
                x.index_fill_(dim, idx, 0)
            else:
                x.index_copy_(dim, idx, x.index_select(dim, src.to(x.device)))
    return {"num_blocks": nb}


def fault_runs(llm, probe, prompts, a):
    """Prefix caching keeps the prompts' blocks resident; corrupt some, rerun, and see who notices."""
    fr = []
    sub = prompts[: a.fault_requests]
    for trial, (mode, nblk) in enumerate([("copy", 4), ("copy", 16), ("zero", 4), ("zero", 16)]):
        _generate(llm, sub, 1)
        info = llm.collective_rpc(corrupt_blocks, args=(nblk, 100 + trial, mode))[0]
        bt, br, fb = score(probe, sub, _generate(llm, sub, a.new_tokens), a.new_tokens)
        llm.reset_prefix_cache()
        ct, _, _ = score(probe, sub, _generate(llm, sub, a.new_tokens), a.new_tokens)
        fr.append({"mode": mode, "blocks": nblk, "cache_blocks": info["num_blocks"], "requests": len(sub),
                   "requests_flagged": br, "median_first_bad_step": float(np.median(fb)) if fb else None,
                   "after_cache_reset_mismatches": ct})
    return fr


def run_entry(entry, probe, a, rng, max_len):
    t0 = time.time()
    llm = _llm(a.probe, max_len, a.new_tokens, entry["kw"])
    t_init = time.time() - t0
    prompts, warm = make_workload(entry, probe, rng, a.requests, max_len, a.new_tokens)
    if warm:
        _generate(llm, prompts[:warm], 1)               # warm the prefix cache with the base prompt
    t1 = time.time()
    outs = _generate(llm, prompts, a.new_tokens, 1.0 if entry.get("sample") else 0.0)
    dt = time.time() - t1
    diag = {}
    details = []
    bad_tok, bad_req, _ = score(probe, prompts, outs, a.new_tokens, diag, details)
    import vllm
    rec = {"config": entry["name"], "engine": "vllm", "engine_version": vllm.__version__, "program": probe.program, "diagnoses": diag, "failures": details, "kwargs": {k: str(v) for k, v in entry["kw"].items()},
           "sample": bool(entry.get("sample")), "requests": len(prompts),
           "prompt_tokens": int(sum(map(len, prompts))), "gen_tokens": len(prompts) * a.new_tokens,
           "mismatched_tokens": bad_tok, "mismatched_requests": bad_req,
           "gen_sec": round(dt, 1), "init_sec": round(t_init, 1)}
    if "speculative_config" in entry["kw"]:
        rec["spec_decode"] = spec_stats(llm)
    if entry.get("faults"):
        try:
            rec["faults"] = fault_runs(llm, probe, prompts, a)
        except Exception as e:
            rec["faults_error"] = repr(e)[:800]
        try:
            from .localize import targeted_campaign
            sub = [p for p in prompts if len(p) <= 4000][:16]
            rec["targeted_faults"] = targeted_campaign(llm, probe, sub, a.new_tokens, rng, _generate)
        except Exception as e:
            rec["targeted_faults_error"] = repr(e)[:800]
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return rec
