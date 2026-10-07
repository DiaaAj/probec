"""SGLang adapter: offline Engine with token-id prompts (skip_tokenizer_init)."""
import collections
import gc
import time

from . import make_workload, score

BACKENDS_DEFAULT = ["flashinfer", "triton", "flex_attention"]
BACKENDS_HOPPER = ["fa3"]


def matrix(name, kind="dense"):
    plan = []
    backends = ["flashinfer", "triton"] if name == "quick" else list(BACKENDS_DEFAULT)
    if name == "full":
        backends += ["torch_native"]
    if name == "hopper":
        backends = ["fa3", "flashinfer", "triton"] if kind == "dense" else ["flashmla", "fa3", "triton"]
    elif kind == "mla":
        backends = ["triton", "flashinfer"]
    for be in backends:
        base = {"attention_backend": be}
        plan.append({"name": f"{be}/default", "kw": dict(base)})
        if name == "quick":
            continue
        plan.append({"name": f"{be}/chunk512_sample", "sample": True,
                     "kw": dict(base, chunked_prefill_size=512)})
        plan.append({"name": f"{be}/noradix_eager_pressure",
                     "kw": dict(base, disable_radix_cache=True, disable_cuda_graph=True,
                                disable_overlap_schedule=True, max_total_tokens=40000)})
        if be in ("flashinfer", "triton", "fa3"):
            plan.append({"name": f"{be}/page16", "kw": dict(base, page_size=16)})
    if name == "ep2":
        return [{"name": "default/ep2_tp2", "kw": {"tp_size": 2, "ep_size": 2}},
                {"name": "default/tp2", "kw": {"tp_size": 2}},
                {"name": "default/ep2_eplb_redundant16", "kw": {"tp_size": 2, "ep_size": 2, "enable_eplb": True,
                                                                "ep_num_redundant_experts": 16}}]
    if name != "quick":
        fp8 = ("flashinfer", "triton")
        if name == "hopper":
            fp8 = ("fa3", "flashinfer", "triton") if kind == "dense" else ("flashmla", "triton")
        for be in fp8:
            kw = {"attention_backend": be, "kv_cache_dtype": "fp8_e4m3"}
            if be == "flashmla":
                kw["mem_fraction_static"] = 0.6        # FP8 FlashMLA allocates a large workspace (37 GiB seen)
            plan.append({"name": f"{be}/fp8kv", "kw": kw})
        plan.append({"name": "flashinfer/ngram_spec",
                     "kw": {"attention_backend": "flashinfer", "speculative_algorithm": "NGRAM",
                            "speculative_num_draft_tokens": 4}})
        plan.append({"name": "flashinfer/ngram_spec_decoys", "workload": "decoys",
                     "kw": {"attention_backend": "flashinfer", "speculative_algorithm": "NGRAM",
                            "speculative_num_draft_tokens": 4}})
    return plan


def _engine(model, max_len, n_new, kw):
    import sglang as sgl
    kw = dict(kw)
    mem = kw.pop("mem_fraction_static", 0.8)
    return sgl.Engine(model_path=model, skip_tokenizer_init=True, dtype="bfloat16", random_seed=0,
                      context_length=max_len + n_new + 128, mem_fraction_static=mem, log_level="error", **kw)


def _generate(engine, prompts, n_new, temperature=0.0, meta=None):
    sp = {"max_new_tokens": n_new, "temperature": temperature, "ignore_eos": True}
    outs = engine.generate(input_ids=prompts, sampling_params=sp)
    if isinstance(outs, dict):
        outs = [outs]
    if meta is not None:
        meta.extend(o.get("meta_info") or {} for o in outs)
    return [list(o.get("output_ids") or o.get("token_ids") or []) for o in outs]


def spec_stats(metas):
    """Per-request speculative-decoding fields of meta_info: counts summed, rates and lengths averaged over
    requests, plus generated tokens per verify step."""
    out, n = {}, collections.Counter()
    for m in metas:
        for k, v in m.items():
            if k.startswith("spec_") and isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
                n[k] += 1
    for k in out:
        if "rate" in k or "length" in k:
            out[k] = round(out[k] / n[k], 4)
    gen = sum(m.get("completion_tokens", 0) for m in metas)
    if out.get("spec_verify_ct"):
        out["tokens_per_verify_step"] = round(gen / out["spec_verify_ct"], 3)
    return out


def run_entry(entry, probe, a, rng, max_len):
    t0 = time.time()
    eng = _engine(a.probe, max_len, a.new_tokens, entry["kw"])
    t_init = time.time() - t0
    try:
        prompts, warm = make_workload(entry, probe, rng, a.requests, max_len, a.new_tokens)
        if warm:
            _generate(eng, prompts[:warm], 1)
        t1 = time.time()
        metas = []
        outs = _generate(eng, prompts, a.new_tokens, 1.0 if entry.get("sample") else 0.0, meta=metas)
        dt = time.time() - t1
        diag = {}
        details = []
        bad_tok, bad_req, _ = score(probe, prompts, outs, a.new_tokens, diag, details)
    finally:
        eng.shutdown()
        gc.collect()
    import sglang
    rec = {"config": entry["name"], "engine": "sglang", "engine_version": getattr(sglang, "__version__", "?"),
           "program": probe.program, "diagnoses": diag, "failures": details, "kwargs": {k: str(v) for k, v in entry["kw"].items()},
           "sample": bool(entry.get("sample")), "requests": len(prompts),
           "prompt_tokens": int(sum(map(len, prompts))), "gen_tokens": len(prompts) * a.new_tokens,
           "mismatched_tokens": bad_tok, "mismatched_requests": bad_req,
           "gen_sec": round(dt, 1), "init_sec": round(t_init, 1)}
    if "speculative_algorithm" in entry["kw"]:
        rec["spec_decode"] = spec_stats(metas)
    return rec
