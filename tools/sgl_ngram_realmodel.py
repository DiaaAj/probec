"""Real-model check: does SGLang NGRAM speculative decoding change greedy outputs of real Qwen3-0.6B?
Prompts repeat token sequences so that n-gram drafts get proposed and accepted."""
import json, sys, time
import numpy as np


def main():
    out, mode = sys.argv[1], sys.argv[2]          # mode: spec | nospec | spec_det | nospec_det
    import sglang as sgl
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download("Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(path)
    rng = np.random.default_rng(0)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
             "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango"]
    prompts = []
    for i in range(96):
        items = [f"{rng.choice(words)}-{int(rng.integers(0, 1000))}" for _ in range(int(rng.integers(20, 120)))]
        text = ("Here is a list of codes: " + ", ".join(items) + ".\nRepeat the list exactly, then repeat it "
                "again in reverse order.\nList: " + ", ".join(items[: len(items) // 3]))
        prompts.append(tok(text)["input_ids"])
    kw = dict(attention_backend="flashinfer")
    if mode.startswith("spec"):
        kw.update(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4)
    if mode.endswith("_det"):
        kw.update(enable_deterministic_inference=True)
    t0 = time.time()
    eng = sgl.Engine(model_path=path, dtype="bfloat16", random_seed=0, mem_fraction_static=0.8,
                     log_level="error", skip_tokenizer_init=True, **kw)
    outs = eng.generate(input_ids=prompts, sampling_params={"max_new_tokens": 256, "temperature": 0.0,
                                                           "ignore_eos": True})
    rec = {"mode": mode, "sec": round(time.time() - t0, 1),
           "outputs": [list(o.get("output_ids") or []) for o in outs]}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(mode, "done", rec["sec"], flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
