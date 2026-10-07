"""Analyze SGLang NGRAM failures: for each failing request, explain the first wrong token against the
recall oracle (what read would have produced it)."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from probec.oracle import Probe                      # noqa: E402
sys.path.insert(0, "tools")
from sgl_ngram_investigate import workload           # noqa: E402


def explain(probe, prompt, exp, got_tok, j):
    """Hypotheses for got_tok at step j (exp = expected continuation)."""
    C = probe.C
    out = []
    for k in range(-6, 7):
        if 0 <= j + k < len(exp) and exp[j + k] == got_tok and k != 0:
            out.append(f"equals expected token of step {j + k} (shift {k:+d})")
    if got_tok >= C and got_tok < 2 * C:
        c = got_tok - C
        prev = prompt[-1] if j == 0 else exp[j - 1]
        start, hops = probe.hops(prev)
        chain = [start]
        for _ in range(probe.L + 2):
            chain.append(int(probe.nxt[chain[-1]]))
        if c in chain:
            out.append(f"class reached after {chain.index(c)} hops from this step's start (expected {hops})")
    else:
        out.append("not an echo token")
    return out


def main():
    probe_dir, results = sys.argv[1], sys.argv[2]
    probe = Probe(probe_dir)
    prompts, shared = workload(probe, np.random.default_rng(13), 192, 16000)
    for line in open(results):
        r = json.loads(line)
        if not r.get("fails"):
            print(r["variant"], "failed", r.get("failed_requests"), r.get("error", ""))
            continue
        print(f"== {r['variant']}: {r['failed_requests']} requests, {r['bad_tokens']} bad tokens")
        for f in r["fails"]:
            p = prompts[f["req"]]
            exp = probe.expected(p, 256)
            j = f["first_bad"]
            got_tok = f["got"][min(3, j)] if f["got"] else None
            print(f"  req {f['req']:3d} len {f['prompt_len']:5d} shared {f['shared_with']} first_bad {j:3d} "
                  f"n_bad {f['n_bad']:3d} meta {f.get('meta')}")
            if got_tok is not None:
                print("     ", explain(probe, p, exp, got_tok, j))


if __name__ == "__main__":
    main()
