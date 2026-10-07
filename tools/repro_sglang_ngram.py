"""Minimal repro: SGLang NGRAM speculative decoding vs no speculation, greedy, on a probec probe.

Greedy speculative decoding must be lossless, so every request's output must be identical with and
without NGRAM. The probe also has an exact oracle, so both runs are checked against it.

    probec compile --like Qwen/Qwen3-0.6B --no-scale-check -o probes/qwen3-0.6b-twin
    python tools/repro_sglang_ngram.py probes/qwen3-0.6b-twin [--overlap/--no-overlap ...]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from probec.oracle import Probe                       # noqa: E402
from sgl_ngram_investigate import workload            # noqa: E402


def run(probe_dir, prompts, spec, extra):
    import sglang as sgl
    kw = dict(model_path=probe_dir, skip_tokenizer_init=True, dtype="bfloat16", random_seed=0,
              context_length=16512, mem_fraction_static=0.8, log_level="error", attention_backend="flashinfer")
    kw.update(extra)
    if spec:
        kw.update(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4)
    eng = sgl.Engine(**kw)
    try:
        outs = eng.generate(input_ids=prompts, sampling_params={"max_new_tokens": 256, "temperature": 0.0,
                                                               "ignore_eos": True})
        return [list(o["output_ids"]) for o in outs]
    finally:
        eng.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--requests", type=int, default=192)
    ap.add_argument("--extra", default="{}", help="extra sgl.Engine kwargs as JSON, e.g. '{\"disable_radix_cache\": true}'")
    a = ap.parse_args()
    probe = Probe(a.probe)
    prompts, _ = workload(probe, np.random.default_rng(13), a.requests, 16000)
    extra = json.loads(a.extra)
    base = run(a.probe, prompts, False, extra)
    spec = run(a.probe, prompts, True, extra)
    diverge = [i for i, (x, y) in enumerate(zip(base, spec)) if x != y]
    wrong = [i for i, (p, y) in enumerate(zip(prompts, spec)) if y != probe.expected(p, 256)]
    print(f"NGRAM vs no-spec: {len(diverge)}/{len(prompts)} requests differ; NGRAM vs exact oracle: {len(wrong)} wrong")
    for i in diverge[:8]:
        j = next(k for k, (x, y) in enumerate(zip(base[i], spec[i])) if x != y)
        print(f"  request {i}: prompt {len(prompts[i])} tokens, first divergence at output token {j}")


if __name__ == "__main__":
    main()
