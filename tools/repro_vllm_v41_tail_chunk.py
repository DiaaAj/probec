"""Reproduce: DeepSeek-V4.1 on vLLM gives wrong greedy output when the last chunk of a chunked prefill is short
enough to run as a PIECEWISE CUDA graph.

Builds probec's DeepSeek-V4.1 probe twin (production widths, 64 experts, ~6 GB, deterministic) from the released
config.json, then sends one prompt of `--length` tokens with `max_num_batched_tokens=--budget` and compares the
greedy output with the twin's known answer.

    python tools/repro_vllm_v41_tail_chunk.py --budget 8192 --length 8202            # last chunk 10: wrong
    python tools/repro_vllm_v41_tail_chunk.py --budget 8192 --length 8202 --eager    # exact
    python tools/repro_vllm_v41_tail_chunk.py --budget 8192 --length 8227            # last chunk 35: wrong
"""
import argparse
import json
import os
import urllib.request

import numpy as np

CONFIG = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--twin", default="dsv41-twin", help="twin directory (built on first use)")
    ap.add_argument("--budget", type=int, default=8192, help="max_num_batched_tokens")
    ap.add_argument("--length", type=int, default=8202, help="prompt length in tokens")
    ap.add_argument("--n", type=int, default=8, help="tokens to generate")
    ap.add_argument("--eager", action="store_true", help="enforce_eager=True (control)")
    ap.add_argument("--compilation-config", default=None, help="JSON passed as compilation_config")
    a = ap.parse_args()

    from probec.dsv41.build import Oracle, compile_twin
    if not os.path.exists(os.path.join(a.twin, "probe_meta.json")):
        print(f"building the twin in {a.twin} (about 6 GB, a few minutes)")
        compile_twin(json.load(urllib.request.urlopen(CONFIG)), a.twin, size="mid", id_offset=1000)
    orc = Oracle(a.twin)
    prompt = orc.prompt(5, a.length, np.random.default_rng(0))      # pool 5: 256 classes, at most one per pair
    expected = orc.expected(prompt, a.n)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=a.twin, skip_tokenizer_init=True, language_model_only=True, max_model_len=a.length + 64,
              max_num_batched_tokens=a.budget, enforce_eager=a.eager, seed=0,
              **({"compilation_config": json.loads(a.compilation_config)} if a.compilation_config else {}))
    out = llm.generate([TokensPrompt(prompt_token_ids=orc.phys(prompt))],
                       SamplingParams(temperature=0.0, max_tokens=a.n, ignore_eos=True, detokenize=False))
    got = orc.logical(out[0].outputs[0].token_ids)
    last = a.length % a.budget or a.budget
    print(f"prompt {a.length} tokens, budget {a.budget} -> last prefill chunk {last} tokens"
          f"{' (eager)' if a.eager else ''}")
    print("expected", expected)
    print("got     ", got)
    print("EXACT" if got == expected else "WRONG")


if __name__ == "__main__":
    main()
