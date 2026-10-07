"""Investigate SGLang NGRAM speculative decoding mismatches on a probe: rerun the exact session-1 workload
under one configuration variant and record every mismatching request (first bad step, context)."""
import json, sys, time
import numpy as np

sys.path.insert(0, "/root/exp/probec/src")
from probec.oracle import Probe          # noqa: E402
from probec.harness import lengths       # noqa: E402

VARIANTS = {
    "A_baseline": dict(attention_backend="flashinfer"),
    "B_noradix": dict(attention_backend="flashinfer", disable_radix_cache=True),
    "C_nooverlap": dict(attention_backend="flashinfer", disable_overlap_schedule=True),
    "D_nocudagraph": dict(attention_backend="flashinfer", disable_cuda_graph=True),
    "E_triton": dict(attention_backend="triton"),
    "F_sequential": dict(attention_backend="flashinfer", max_running_requests=1),
    "G_draft2": dict(attention_backend="flashinfer", speculative_num_draft_tokens=2),
    "H_nospec": dict(attention_backend="flashinfer"),
    "I_noshared_prompts": dict(attention_backend="flashinfer"),
}


def workload(probe, rng, n_req, max_len, shared_frac=0.3, allow_shared=True):
    """Same RNG sequence as probec.harness.workload, also returning which prompts share a prefix."""
    prompts, shared = [], []
    for n in lengths(rng, n_req, max_len):
        if prompts and rng.random() < shared_frac and allow_shared:
            base = int(rng.integers(0, len(prompts)))
            prompts.append(probe.shared_variant(prompts[base], rng))
            shared.append(base)
        else:
            prompts.append(probe.sample_prompt(rng, n))
            shared.append(None)
    return prompts, shared


def main():
    probe_dir, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]
    import sglang as sgl
    probe = Probe(probe_dir)
    kw = dict(VARIANTS[variant])
    spec = {}
    if variant != "H_nospec":
        spec = {"speculative_algorithm": "NGRAM",
                "speculative_num_draft_tokens": kw.pop("speculative_num_draft_tokens", 4)}
    max_len = 16000
    prompts, shared = workload(probe, np.random.default_rng(13), 192, max_len,
                               allow_shared=variant != "I_noshared_prompts")
    t0 = time.time()
    eng = sgl.Engine(model_path=probe_dir, skip_tokenizer_init=True, dtype="bfloat16", random_seed=0,
                     context_length=max_len + 512, mem_fraction_static=0.8, log_level="error", **kw, **spec)
    outs = eng.generate(input_ids=prompts, sampling_params={"max_new_tokens": 256, "temperature": 0.0,
                                                           "ignore_eos": True})
    dt = time.time() - t0
    fails = []
    for i, (p, o) in enumerate(zip(prompts, outs)):
        got = list(o.get("output_ids") or [])
        exp = probe.expected(p, 256)
        d = [j for j, (a, b) in enumerate(zip(got, exp)) if a != b]
        if d or len(got) != 256:
            j = d[0] if d else len(got)
            fails.append({"req": i, "prompt_len": len(p), "shared_with": shared[i], "first_bad": j,
                          "n_bad": len(d), "got": got[max(0, j - 3):j + 4], "exp": exp[max(0, j - 3):j + 4],
                          "got_len": len(got),
                          "meta": {k: v for k, v in (o.get("meta_info") or {}).items()
                                   if k in ("spec_verify_ct", "completion_tokens", "spec_accept_length",
                                            "cached_tokens", "prompt_tokens")}})
    rec = {"variant": variant, "kw": {k: str(v) for k, v in {**kw, **spec}.items()}, "requests": len(prompts),
           "failed_requests": len(fails), "bad_tokens": sum(f["n_bad"] for f in fails), "sec": round(dt, 1),
           "fails": fails}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "fails"}), flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
