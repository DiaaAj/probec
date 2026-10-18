"""The same request, alone, many times: does SGLang give the same (exact) output every time?

Runs prompts from a probec twin one at a time (max_running_requests=1) through one SGLang Engine, each
--reps times, and compares every output with the oracle. A correct engine gives the oracle's tokens on every
repeat; a race or an uninitialized buffer shows as repeats that differ.

    python tools/sgl_repeat.py <twin> [--prompts 4] [--reps 32] [--len 1500] [--new 256] [--kw '{"attention_backend": "flashinfer"}']
"""
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from probec.oracle import Probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--len", type=int, default=1500)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--kw", default='{"attention_backend": "flashinfer"}')
    a = ap.parse_args()
    import sglang as sgl
    probe = Probe(a.twin)
    rng = np.random.default_rng(11)
    prompts = [probe.sample_prompt(rng, a.len) for _ in range(a.prompts)]
    eng = sgl.Engine(model_path=a.twin, skip_tokenizer_init=True, dtype="bfloat16", random_seed=0,
                     context_length=max(map(len, prompts)) + a.new + 128, mem_fraction_static=0.8,
                     log_level="error", max_running_requests=1, **json.loads(a.kw))
    sp = {"max_new_tokens": a.new, "temperature": 0.0, "ignore_eos": True}
    total_bad = 0
    try:
        for i, p in enumerate(prompts):
            exp = probe.expected(p, a.new)
            outcomes = collections.Counter()
            first_bad = []
            for _ in range(a.reps):
                o = eng.generate(input_ids=[p], sampling_params=sp)
                o = o[0] if isinstance(o, list) else o
                got = list(o.get("output_ids") or o.get("token_ids") or [])
                d = [j for j, (x, y) in enumerate(zip(got, exp)) if x != y]
                outcomes[tuple(got)] += 1
                if d:
                    first_bad.append(d[0])
            bad = len(first_bad)
            total_bad += bad
            print(f"REPEAT prompt {i} ({len(p)} tokens): {bad}/{a.reps} repeats wrong, {len(outcomes)} distinct "
                  f"outputs; first wrong steps {sorted(first_bad)[:12]}", flush=True)
    finally:
        eng.shutdown()
    print(f"REPEAT total: {total_bad}/{a.prompts * a.reps} runs wrong", flush=True)


if __name__ == "__main__":
    main()
