"""One probec workload through SGLang with chosen Engine kwargs: is a mismatch deterministic, and which
setting removes it? Uses the workload of the matrix entry at index `--seed` (0 = `flashinfer/default`), so
a failing entry can be rerun with one setting changed. Run each variant in its own process.

    python tools/sgl_variant.py <twin> <out.jsonl> <name> '<Engine kwargs as JSON>' [--seed 0]
"""
import argparse
import json
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, "src")
from probec.harness import sglang_engine        # noqa: E402
from probec.oracle import Probe                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("out")
    ap.add_argument("name")
    ap.add_argument("kwargs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probe = Probe(a.twin)
    args = NS(probe=a.twin, requests=192, new_tokens=256, max_len=16000, fault_requests=64)
    max_len = min(args.max_len, probe.max_context - args.new_tokens - 64)
    entry = {"name": a.name, "kw": json.loads(a.kwargs)}
    rec = sglang_engine.run_entry(entry, probe, args, np.random.default_rng(a.seed), max_len)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    fails = [(x["request"], x["first_bad_step"], x["n_bad"]) for x in rec.get("failures", [])]
    print(f"VARIANT {a.name}: {rec['mismatched_tokens']} mismatched tokens in {rec['mismatched_requests']} requests "
          f"(request, first bad step, bad tokens): {fails[:6]}")


if __name__ == "__main__":
    main()
