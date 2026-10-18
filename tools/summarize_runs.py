"""Summarize GPU validation runs: generated tokens checked per session and engine, and every entry with
mismatches, classified by the known causes recorded in PROGRESS.md.

usage: python tools/summarize_runs.py [runs_dir]
"""
import collections
import glob
import json
import os
import sys

HARDWARE = {"m1-gpu1": "RTX 3090", "m2-gpu2": "RTX 3090", "s3-gpu3": "RTX 3090", "s4-replay": "RTX 3090",
            "s5-ep": "2x RTX 3090", "s6-hopper": "H100", "s7": "RTX 3090", "s9a2": "RTX 3090",
            "s9b2": "RTX 3090", "s10": "H100", "s11": "H100", "s12": "RTX 3090", "s13c": "B200",
            "s14": "B200", "s16": "RTX 3090", "s19": "RTX 3090", "s20": "H100 PCIe", "s21": "RTX 3090",
            "s22": "B200", "s23b": "RTX 3090", "s24": "RTX 3090", "s25": "RTX 3090"}
# probe-side causes, fixed during validation: the positional probe before the identical-token prefix
# (sessions 6 and 7), and the first MLA scale check under FP8 query quantization (session 10)
PROBE_BUGS = {("s6-hopper", "posrecall"), ("s7", "posrecall"), ("s10", "mla", "FLASHMLA/fp8kv")}


def cause(session, f, r):
    prog, cfg = r.get("program"), r.get("config")
    if (session, prog) in PROBE_BUGS or (session, prog, cfg) in PROBE_BUGS:
        return "probe bug (fixed)"
    if r.get("engine") == "sglang" and "ngram" in (cfg or ""):
        return "engine bug: SGLang NGRAM + radix cache + overlap + CUDA graphs"
    # records before engine_version existed: the release is known from the file (replay_<ver>, bisect)
    # or, in session 12, from the job log (only the 0.27.1 control failed)
    old = r.get("engine_version", "").startswith(("0.25", "0.26", "0.27"))
    old = old or "replay_0.2" in f or "bisect" in f or (session == "s12" and "engine_version" not in r)
    if prog == "mla" and old and cfg in ("default/prefix", "mla/big_budget_many_seqs"):
        return "engine bug: vLLM MLA prefix caching (<= 0.27.1, fixed by #50613 in 0.28.0)"
    if prog == "mla" and cfg == "default/shared_prefix_long_suffix":
        return "engine bug: vLLM #42642 (0.25.1)"
    split = (cfg or "").startswith("default/split_markers")
    if prog == "mla" and split and r.get("engine_version") == "0.28.0":
        return "engine bug: vLLM #52796 (0.28.0, Blackwell MLA prefill LSE base)"
    name = os.path.basename(f)
    if session == "s24" and name.startswith("0.23.0_8b_fp8"):
        return "engine bug: vLLM #44735 (<= 0.23.0, FP8 Marlin skips the transpose of square layers)"
    if session == "s24" and name.startswith("0.18.1_") and "unit" in name:
        return "engine bug: vLLM #34577 (<= 0.18.1, NVFP4 Marlin scale underflow)"
    if r.get("engine") == "sglang" and "nvfp4" in name and "-mo" not in name and "modelopt" not in name \
            and session in ("s20", "s21", "s23b"):
        return "engine bug: SGLang NVFP4 Marlin scale underflow (vLLM #34577 not ported; SM80-SM90)"
    sgl_ampere = r.get("engine") == "sglang" and session in ("s21", "s23b")
    if sgl_ampere and ("fp8-block" in name or "fp8block" in name):
        return ("engine bug (under investigation): SGLang 0.5.22 FP8 Marlin on SM86, a few requests per run, "
                "nondeterministic, needs the overlap scheduler")
    if sgl_ampere and ("int4-awq" in name or "moeawq" in name):
        return ("engine bug (under investigation): SGLang 0.5.22 Marlin MoE (AWQ) on SM86, wrong expert outputs "
                "in some decode steps, nondeterministic")
    return "UNEXPLAINED"


def main(root):
    tok = collections.Counter()
    ent = collections.Counter()
    bad = []
    for f in sorted(glob.glob(os.path.join(root, "*", "out", "*.jsonl"))):
        session = f.split(os.sep)[-3]
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            g = r.get("gen_tokens")
            if not isinstance(g, int):
                continue
            key = (session, r.get("engine") or ("sglang" if "sgl" in os.path.basename(f) else "vllm"))
            tok[key] += g
            ent[key] += 1
            if r.get("mismatched_tokens"):
                bad.append((session, os.path.basename(f), r.get("program"), r.get("config"),
                            r["mismatched_tokens"], r.get("mismatched_requests"), g, cause(session, f, r)))
    print("| session | hardware | engine | entries | generated tokens |")
    print("|---|---|---|---|---|")
    for (s, e) in sorted(tok):
        print(f"| {s} | {HARDWARE.get(s, '?')} | {e} | {ent[(s, e)]} | {tok[(s, e)]:,} |")
    total = sum(tok.values())
    bad_gen = sum(b[6] for b in bad)
    print(f"\ntotal: {sum(ent.values())} entries, {total:,} generated tokens; "
          f"{len(bad)} entries with mismatches ({bad_gen:,} tokens in them)")
    by = collections.Counter(b[7] for b in bad)
    for c, n in by.most_common():
        print(f"  {n:3d} entries: {c}")
    print("\nentries with mismatches:")
    for b in bad:
        print(f"  {b[0]:10s} {b[1]:28s} {str(b[2]):9s} {str(b[3]):36s} bad {b[4]:6d} req {b[5]}  -> {b[7]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
