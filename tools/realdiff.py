"""Real-model differential testing next to probec, on the same engine configurations (docs/REPORT.md §1).

A real model can only be checked by comparing its outputs across configurations, or with HF
transformers. A probe twin is checked against its oracle. For every configuration this runs both, each
in its own process:
  - side "real": greedy outputs of the real model on real-text prompts (one shared document per eight
    requests, with tasks that copy from it, so prefix caching and n-gram drafts are exercised), with the
    top-5 logprobs of every generated token;
  - side "probe": the twin's outputs on probe prompts, scored against the oracle.
Both sides request the top-5 logprobs, as engines' own logprob-comparison tests do; for speculative
configurations they also run without logprobs (sides "real0", "probe0"), in case logprobs change the path.
Records go to <out>/<tag>__<side>__<config>.json.gz (outputs included) and a summary line per record to
<out>/<tag>.jsonl. tools/realdiff_analyze.py compares them.

    python tools/realdiff.py --engine vllm --model Qwen/Qwen3-0.6B --twin probes/dense --tag qwen3-0.6b --out out/realdiff
    python tools/realdiff.py --engine hf --model Qwen/Qwen3-0.6B --tag qwen3-0.6b --out out/realdiff
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

NGRAM = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
CONFIGS = {
    "vllm": [("FLASH_ATTN/default", {"attention_backend": "FLASH_ATTN"}),
             ("FLASH_ATTN/rerun", {"attention_backend": "FLASH_ATTN"}),
             ("FLASHINFER/default", {"attention_backend": "FLASHINFER"}),
             ("TRITON_ATTN/default", {"attention_backend": "TRITON_ATTN"}),
             ("FLASH_ATTN/no_prefix_cache", {"attention_backend": "FLASH_ATTN", "enable_prefix_caching": False}),
             ("FLASH_ATTN/chunk512", {"attention_backend": "FLASH_ATTN", "max_num_batched_tokens": 512}),
             ("FLASH_ATTN/eager", {"attention_backend": "FLASH_ATTN", "enforce_eager": True}),
             ("FLASH_ATTN/one_seq", {"attention_backend": "FLASH_ATTN", "max_num_seqs": 1}),
             ("FLASH_ATTN/ngram", {"attention_backend": "FLASH_ATTN", "speculative_config": NGRAM})],
    "sglang": [("flashinfer/default", {"attention_backend": "flashinfer"}),
               ("flashinfer/rerun", {"attention_backend": "flashinfer"}),
               ("triton/default", {"attention_backend": "triton"}),
               ("flashinfer/no_radix", {"attention_backend": "flashinfer", "disable_radix_cache": True}),
               ("flashinfer/chunk512", {"attention_backend": "flashinfer", "chunked_prefill_size": 512}),
               ("flashinfer/no_cuda_graph", {"attention_backend": "flashinfer", "disable_cuda_graph": True}),
               ("flashinfer/no_overlap", {"attention_backend": "flashinfer", "disable_overlap_schedule": True}),
               ("flashinfer/one_seq", {"attention_backend": "flashinfer", "max_running_requests": 1}),
               ("flashinfer/ngram", {"attention_backend": "flashinfer", "speculative_algorithm": "NGRAM",
                                     "speculative_num_draft_tokens": 4})],
    "hf": [("HF/generate", {})],
}

ITEMS = ["copper fittings", "steel brackets", "nylon washers", "brass valves", "rubber gaskets", "aluminum rails",
         "ceramic insulators", "PVC elbows", "zinc anchors", "glass fuses", "oak dowels", "carbon brushes",
         "silicone seals", "titanium screws", "cotton filters", "lithium cells"]
SITES = ["Warehouse A", "Warehouse B", "Warehouse C", "Dock 4", "Dock 7", "North depot", "South depot", "Rail yard"]
STATUS = ["delivered", "in transit", "delayed", "returned", "awaiting pickup"]
TASKS = ["Repeat the log exactly as written.",
         "List the order numbers of all delivered shipments, one per line.",
         "Rewrite the first ten entries as full sentences.",
         "Copy every entry from Warehouse B, unchanged.",
         "Repeat the log in reverse order.",
         "Summarize the log in one short paragraph.",
         "Convert the log to CSV with the header date,order,quantity,item,site,status.",
         "Continue the log with ten more entries in the same format."]


def documents(rng, n):
    docs = []
    for _ in range(n):
        lines = []
        for _ in range(int(rng.integers(30, 110))):
            lines.append(f"2025-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d} | order "
                         f"{int(rng.integers(10000, 100000))} | {int(rng.integers(1, 200))} x "
                         f"{ITEMS[int(rng.integers(len(ITEMS)))]} | {SITES[int(rng.integers(len(SITES)))]} | "
                         f"{STATUS[int(rng.integers(len(STATUS)))]}")
        docs.append("\n".join(lines))
    return docs


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def real_prompts(tok, n_req, seed=0):
    """Eight tasks per shared document, consecutive requests sharing it (prefix-cache hits while they run)."""
    rng = np.random.default_rng(seed)
    docs = documents(rng, (n_req + len(TASKS) - 1) // len(TASKS))
    return [chat_ids(tok, f"Shipment log:\n{docs[i // len(TASKS)]}\n\n{TASKS[i % len(TASKS)]}") for i in range(n_req)]


def stop_ids(tok, path):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    try:
        from transformers import GenerationConfig
        e = GenerationConfig.from_pretrained(path).eos_token_id
        ids.update([e] if isinstance(e, int) else list(e or []))
    except Exception:
        pass
    return sorted(int(i) for i in ids)


def digest(prompts):
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:16]


def is_multimodal(path):
    with open(os.path.join(path, "config.json")) as f:
        return any("ConditionalGeneration" in a for a in json.load(f).get("architectures", []))


# ---- engines ----------------------------------------------------------------------------------------------

def vllm_generate(llm, prompts, n_new, stops, logprobs):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_new, temperature=0.0, detokenize=False, logprobs=logprobs or None,
                        stop_token_ids=stops or None, ignore_eos=not stops)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]
        top = None
        if logprobs and c.logprobs:
            top = [[[int(k), round(float(v.logprob), 3)] for k, v in sorted(d.items(), key=lambda kv: -kv[1].logprob)[:logprobs]]
                   for d in c.logprobs]
        res.append({"ids": [int(t) for t in c.token_ids], "top": top})
    return res


def sglang_generate(eng, prompts, n_new, stops, logprobs, metas=None):
    sp = {"max_new_tokens": n_new, "temperature": 0.0}
    if stops:
        sp["stop_token_ids"] = stops
    else:
        sp["ignore_eos"] = True
    kw = {"return_logprob": True, "top_logprobs_num": logprobs, "logprob_start_len": -1} if logprobs else {}
    try:
        outs = eng.generate(input_ids=prompts, sampling_params=sp, **kw)
    except TypeError:
        kw.pop("logprob_start_len", None)
        outs = eng.generate(input_ids=prompts, sampling_params=sp, **kw)
    if isinstance(outs, dict):
        outs = [outs]
    res = []
    for o in outs:
        m = o.get("meta_info") or {}
        if metas is not None:
            metas.append(m)
        top = None
        if logprobs and m.get("output_top_logprobs"):
            top = [[[int(t[1]), round(float(t[0]), 3)] for t in step[:logprobs]] for step in m["output_top_logprobs"]]
        res.append({"ids": [int(t) for t in (o.get("output_ids") or o.get("token_ids") or [])], "top": top})
    return res


def hf_generate(path, prompts, n_new, stops, logprobs, bs=16):
    import torch
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    pad = stops[0]
    res = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[r, L - len(prompts[i]):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), max_new_tokens=n_new, do_sample=False,
                                 eos_token_id=stops, pad_token_id=pad, output_logits=True, return_dict_in_generate=True)
        tops = [torch.log_softmax(x.float(), -1).topk(logprobs, -1) for x in out.logits]
        seqs = out.sequences[:, L:].tolist()
        for r, i in enumerate(idx):
            toks = seqs[r]
            cut = next((j + 1 for j, t in enumerate(toks) if t in stops), len(toks))
            res[i] = {"ids": toks[:cut],
                      "top": [[[int(tops[j].indices[r, q]), round(float(tops[j].values[r, q]), 3)] for q in range(logprobs)]
                              for j in range(cut)]}
    return res


# ---- one configuration, one side (child process) ---------------------------------------------------------

def run_child(a):
    name, kw = dict((n, (n, k)) for n, k in CONFIGS[a.engine])[a.config]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    lp = 0 if a.side.endswith("0") else a.logprobs
    rec = {"tag": a.tag, "side": a.side, "engine": a.engine, "config": name, "logprobs": lp,
           "kwargs": {k: str(v) for k, v in kw.items()}}
    t0 = time.time()
    metas = []
    if a.side.startswith("real"):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        path = snapshot_download(a.model)
        tok = AutoTokenizer.from_pretrained(path)
        prompts = real_prompts(tok, a.requests)
        stops = stop_ids(tok, path)
        rec.update(model=a.model, prompt_digest=digest(prompts), stops=stops, prompt_tokens=int(sum(map(len, prompts))))
        if a.engine == "hf":
            outs = hf_generate(path, prompts, a.new_tokens, stops, lp or 5, bs=a.hf_batch)
            import transformers
            rec["engine_version"] = transformers.__version__
        elif a.engine == "vllm":
            from vllm import LLM
            import vllm
            extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if is_multimodal(path) else {}
            if "speculative_config" in kw:
                extra["disable_log_stats"] = False
            llm = LLM(model=path, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
                      max_model_len=max(map(len, prompts)) + a.new_tokens + 64, **extra, **kw)
            t1 = time.time()
            outs = vllm_generate(llm, prompts, a.new_tokens, stops, lp)
            rec["gen_sec"] = round(time.time() - t1, 1)
            rec["engine_version"] = vllm.__version__
            if "speculative_config" in kw:
                from probec.harness.vllm_engine import spec_stats
                rec["spec_decode"] = spec_stats(llm)
        else:
            import sglang as sgl
            eng = sgl.Engine(model_path=path, dtype="bfloat16", random_seed=0, skip_tokenizer_init=True,
                             context_length=max(map(len, prompts)) + a.new_tokens + 64, mem_fraction_static=0.8,
                             log_level="error", **kw)
            try:
                t1 = time.time()
                outs = sglang_generate(eng, prompts, a.new_tokens, stops, lp, metas)
                rec["gen_sec"] = round(time.time() - t1, 1)
            finally:
                eng.shutdown()
            rec["engine_version"] = getattr(sgl, "__version__", "?")
        rec["outputs"] = outs
    else:
        from probec.harness import make_workload, score
        from probec.oracle import Probe
        probe = Probe(a.twin)
        max_len = min(16000, probe.max_context - a.new_tokens - 64)
        rng = np.random.default_rng(a.probe_seed)
        spec = "speculative_config" in kw or "speculative_algorithm" in kw
        workload = "decoys" if spec and a.engine == "vllm" else None
        prompts, warm = make_workload({"workload": workload}, probe, rng, a.requests, max_len, a.new_tokens)
        if a.engine == "vllm":
            from probec.harness import vllm_engine as ve
            import vllm
            llm = ve._llm(a.twin, max_len, a.new_tokens, dict(kw, gpu_memory_utilization=0.85))
            if warm:
                vllm_generate(llm, prompts[:warm], 1, [], 0)
            t1 = time.time()
            outs = vllm_generate(llm, prompts, a.new_tokens, [], lp)
            rec["gen_sec"] = round(time.time() - t1, 1)
            rec["engine_version"] = vllm.__version__
            if spec:
                rec["spec_decode"] = ve.spec_stats(llm)
        else:
            from probec.harness import sglang_engine as se
            import sglang as sgl
            eng = se._engine(a.twin, max_len, a.new_tokens, kw)
            try:
                if warm:
                    sglang_generate(eng, prompts[:warm], 1, [], 0)
                t1 = time.time()
                outs = sglang_generate(eng, prompts, a.new_tokens, [], lp, metas)
                rec["gen_sec"] = round(time.time() - t1, 1)
            finally:
                eng.shutdown()
            rec["engine_version"] = getattr(sgl, "__version__", "?")
            if spec:
                rec["spec_decode"] = se.spec_stats(metas)
        details = []
        bad_tok, bad_req, _ = score(probe, prompts, [o["ids"] for o in outs], a.new_tokens, {}, details)
        rec.update(twin=os.path.basename(a.twin.rstrip("/")), program=probe.program, probe_seed=a.probe_seed,
                   workload=workload or "default", prompt_digest=digest(prompts),
                   mismatched_tokens=bad_tok, mismatched_requests=bad_req, failures=details,
                   outputs=[{"ids": o["ids"]} for o in outs])
    if a.engine == "sglang" and metas and "spec_decode" not in rec and "speculative_algorithm" in kw:
        from probec.harness.sglang_engine import spec_stats
        rec["spec_decode"] = spec_stats(metas)
    rec["requests"] = len(prompts)
    rec["sec"] = round(time.time() - t0, 1)
    fname = os.path.join(a.out, f"{a.tag}__{a.side}__{name.replace('/', '_')}.json.gz")
    with gzip.open(fname, "wt") as f:
        json.dump(rec, f)
    summary = {k: v for k, v in rec.items() if k not in ("outputs", "failures")}
    summary["file"] = os.path.basename(fname)
    with open(a.record, "w") as f:
        f.write(json.dumps(summary) + "\n")
    print("REALDIFF", json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(CONFIGS), required=True)
    ap.add_argument("--model", required=True, help="HF id of the real model")
    ap.add_argument("--twin", help="probe twin of the model (side 'probe')")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sides", default="real,probe")
    ap.add_argument("--configs", default="", help="comma-separated subset of configuration names")
    ap.add_argument("--requests", type=int, default=192)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--logprobs", type=int, default=5)
    ap.add_argument("--probe-seed", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--hf-batch", type=int, default=16, help="HF transformers batch size")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--side")
    ap.add_argument("--config")
    ap.add_argument("--record")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.child:
        return run_child(a)
    from probec.harness import run_isolated
    names = [n for n, _ in CONFIGS[a.engine] if not a.configs or n in a.configs.split(",")]
    sides = [s for s in a.sides.split(",") if s and not (s.startswith("probe") and (a.engine == "hf" or not a.twin))]
    spec = {n for n, kw in CONFIGS[a.engine] if "speculative_config" in kw or "speculative_algorithm" in kw}
    jobs = [f"{s}|{n}" for s in sides for n in names]
    jobs += [f"{s}0|{n}" for s in sides for n in names if n in spec]     # the same without logprobs

    def cmd(job, path):
        s, n = job.split("|")
        argv = [sys.executable, os.path.abspath(__file__), "--child", "--side", s, "--config", n, "--record", path,
                "--engine", a.engine, "--model", a.model, "--tag", a.tag, "--out", a.out,
                "--requests", str(a.requests), "--new-tokens", str(a.new_tokens), "--logprobs", str(a.logprobs),
                "--probe-seed", str(a.probe_seed), "--hf-batch", str(a.hf_batch)]
        return argv + (["--twin", a.twin] if a.twin else [])

    run_isolated(jobs, cmd, os.path.join(a.out, f"{a.tag}.jsonl"), a.timeout, extra={"tag": a.tag, "engine": a.engine})


if __name__ == "__main__":
    main()
