"""Minimal repro: SGLang NGRAM speculative decoding vs no speculation, greedy, on a probec probe.

Greedy speculative decoding must be lossless, so every request's output must be identical with and
without NGRAM. The probe also has an exact oracle, so both runs are checked against it.

    probec compile --like Qwen/Qwen3-0.6B --no-scale-check -o probes/qwen3-0.6b-twin
    python tools/repro_sglang_ngram.py probes/qwen3-0.6b-twin [--finish stop] [--extra '{...}']

How requests end (--finish):
- length (the default): every request generates 256 tokens with ignore_eos.
- stop: each request gets stop_token_ids=[t], where t is the oracle's token at a random output position k in
  [8, 240] that does not occur earlier in its answer, and ignore_eos is off, so the model's EOS ids end
  requests too. A request for which no such k turns up keeps the length ending. The oracle answer is cut at
  the first stop or EOS token. SGLang keeps the matched token in output_ids (Req.finished_len is the matched
  position + 1); the run without speculation decides the convention, and the script reports it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from probec.oracle import Probe                       # noqa: E402
from sgl_ngram_investigate import workload            # noqa: E402

N_NEW = 256
LENGTH_PARAMS = {"max_new_tokens": N_NEW, "temperature": 0.0, "ignore_eos": True}
STOP_SEED = 0                    # stop positions use their own RNG; the workload's (seed 13) is untouched
STOP_RANGE = (8, 240)            # inclusive
STOP_TRIES = 16
META_KEYS = ("finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens", "spec_verify_ct",
             "spec_accept_length")


def run(probe_dir, prompts, spec, extra, sampling_params=LENGTH_PARAMS):
    """Output ids and selected meta_info per request; sampling_params is one dict or one per request."""
    import sglang as sgl
    kw = dict(model_path=probe_dir, skip_tokenizer_init=True, dtype="bfloat16", random_seed=0,
              context_length=16512, mem_fraction_static=0.8, log_level="error", attention_backend="flashinfer")
    kw.update(extra)
    if spec:
        kw.update(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4)
    eng = sgl.Engine(**kw)
    try:
        outs = eng.generate(input_ids=prompts, sampling_params=sampling_params)
        return ([list(o["output_ids"]) for o in outs],
                [{k: v for k, v in (o.get("meta_info") or {}).items() if k in META_KEYS} for o in outs])
    finally:
        eng.shutdown()


def eos_ids(probe_dir):
    """The ids SGLang ends a request on when ignore_eos is off (ModelConfig._get_hf_eos_token_id): the
    eos_token_id of config.json and of generation_config.json, each an int or a list."""
    ids = set()
    for name in ("config.json", "generation_config.json"):
        path = os.path.join(probe_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                v = json.load(f).get("eos_token_id")
            if v is not None:
                ids |= {int(v)} if isinstance(v, int) else {int(t) for t in v}
    return ids


def stop_positions(answers, rng, lo=STOP_RANGE[0], hi=STOP_RANGE[1], tries=STOP_TRIES):
    """Per request, a random output position k in [lo, hi] whose oracle token does not occur earlier in the
    answer, so stop_token_ids=[answer[k]] ends the request exactly there; None if `tries` picks found none."""
    ks = []
    for ans in answers:
        k = None
        for _ in range(tries):
            c = int(rng.integers(lo, hi + 1))
            if c < len(ans) and ans[c] not in ans[:c]:
                k = c
                break
        ks.append(k)
    return ks


def first_stop(answer, stops):
    """Index of the first token of `answer` that is in `stops` (where generation ends), or None."""
    return next((j for j, t in enumerate(answer) if t in stops), None)


def first_diff(x, y):
    """First output index where x and y differ; the shorter length if one is a prefix of the other."""
    return next((j for j, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--requests", type=int, default=192)
    ap.add_argument("--extra", default="{}", help="extra sgl.Engine kwargs as JSON, e.g. '{\"disable_radix_cache\": true}'")
    ap.add_argument("--finish", choices=("length", "stop"), default="length",
                    help="length: 256 tokens per request with ignore_eos (the original repro); stop: each request "
                         "ends on its own stop token id, with ignore_eos off")
    ap.add_argument("--dump", help="also write prompt metadata, the oracle answers and both runs' outputs as JSON")
    a = ap.parse_args()
    probe = Probe(a.probe)
    prompts, shared = workload(probe, np.random.default_rng(13), a.requests, 16000)
    extra = json.loads(a.extra)
    expected = [probe.expected(p, N_NEW) for p in prompts]
    n = len(prompts)
    params, ks, ends, eos = LENGTH_PARAMS, [None] * n, [None] * n, set()
    if a.finish == "stop":
        eos = eos_ids(a.probe)
        ks = stop_positions(expected, np.random.default_rng(STOP_SEED))
        params = [dict(LENGTH_PARAMS) if k is None else
                  {"max_new_tokens": N_NEW, "temperature": 0.0, "ignore_eos": False, "stop_token_ids": [exp[k]]}
                  for exp, k in zip(expected, ks)]
        # where a correct engine ends each answer: its stop token or an earlier EOS id (ignore_eos is off)
        ends = [None if k is None else first_stop(exp, eos | {exp[k]}) for exp, k in zip(expected, ks)]
    base, base_meta = run(a.probe, prompts, False, extra, params)
    spec, spec_meta = run(a.probe, prompts, True, extra, params)
    if a.dump:      # written before the comparison; "expected" holds the full 256-token oracle answers
        with open(a.dump, "w") as f:
            json.dump({"finish": a.finish, "extra": extra, "prompt_lens": [len(p) for p in prompts],
                       "shared_with": shared, "stop_pos": ks, "end_pos": ends, "eos_ids": sorted(eos),
                       "expected": expected, "no_spec": base, "ngram": spec, "no_spec_meta": base_meta,
                       "ngram_meta": spec_meta}, f, default=str)
    convention, stopped, n_incl, n_excl, neither = None, [], 0, 0, []
    if a.finish == "stop":
        stopped = [i for i, j in enumerate(ends) if j is not None]
        n_incl = sum(base[i] == expected[i][:ends[i] + 1] for i in stopped)
        n_excl = sum(base[i] == expected[i][:ends[i]] for i in stopped)
        neither = [i for i in stopped if base[i] not in (expected[i][:ends[i] + 1], expected[i][:ends[i]])]
        # SGLang's source keeps the matched token (the default); the run without speculation can overrule it
        convention = "excluding" if n_excl > n_incl else "including"
        keep = 1 if convention == "including" else 0
        expected = [exp if j is None else exp[:j + keep] for exp, j in zip(expected, ends)]
    diverge = [i for i, (x, y) in enumerate(zip(base, spec)) if x != y]
    wrong = [i for i, (e, y) in enumerate(zip(expected, spec)) if y != e]
    base_wrong = [i for i, (e, x) in enumerate(zip(expected, base)) if x != e]
    print(f"NGRAM vs no-spec: {len(diverge)}/{len(prompts)} requests differ; NGRAM vs exact oracle: {len(wrong)} wrong")
    print(f"no-spec vs exact oracle: {len(base_wrong)} wrong; NGRAM wrong requests: {wrong}")
    if a.finish == "stop":
        kk = [k for k in ks if k is not None]
        pos = (f"stop position min {min(kk)} / median {float(np.median(kk)):g} / max {max(kk)}" if kk
               else "no stop positions")
        early = sum(j is not None and j < k for j, k in zip(ends, ks) if k is not None)
        print(f"stop mode: {len(kk)}/{n} requests end on a stop token, {n - len(kk)} on length; {pos}; "
              f"EOS ids {sorted(eos)} end {early} oracle answers before their stop token")
        where = "included in" if convention == "including" else "excluded from"
        print(f"stop mode: the matched stop token is {where} output_ids (no-spec run: {n_incl} requests match "
              f"including it, {n_excl} excluding it, {len(neither)} neither, of {len(stopped)})")
        if neither:
            print(f"stop mode: the no-spec output matches neither convention for requests {neither}")
        for i in wrong:
            k = "-" if ks[i] is None else ks[i]
            reused = [r for r, s in enumerate(shared) if s == i]
            print(f"  NGRAM wrong {i}: first wrong output token {first_diff(spec[i], expected[i])}, "
                  f"length {len(spec[i])} (oracle {len(expected[i])}), stop position {k}; prompt "
                  + ("is new" if shared[i] is None else f"extends request {shared[i]}'s")
                  + (f", reused by requests {reused}" if reused else ""))
    for i in diverge[:8]:
        j = first_diff(base[i], spec[i])
        tail = "" if a.finish == "length" else f" (lengths: no-spec {len(base[i])}, NGRAM {len(spec[i])})"
        print(f"  request {i}: prompt {len(prompts[i])} tokens, first divergence at output token {j}{tail}")


if __name__ == "__main__":
    main()
