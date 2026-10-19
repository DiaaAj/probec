"""Collect the release-sweep records (one `probec run` output per release and twin, named <version>__<twin>.jsonl)
into docs/data/sweep.csv: one row per (engine, version, twin, entry), with wrong requests or the error.

    python tools/sweep_analyze.py out/sweep [more directories ...]
"""
import csv
import glob
import json
import os
import re
import sys


UNSUPPORTED = ("does not recognize this architecture", "not supported for now", "Invalid type of HuggingFace config",
               "no SGlang implementation", "qwen3_5_text", "are not supported", "is not supported")


def short_error(r):
    """'unsupported: ...' when the release cannot load the twin's architecture, else 'crash: ...'."""
    e = (r.get("error") or "") + " " + (r.get("output_tail") or "")
    if not e.strip():
        return ""
    if any(u in e for u in UNSUPPORTED):
        return "unsupported: the release cannot load this architecture"
    for pat, msg in ((r"Ninja build failed", "crash: a FlashInfer kernel failed to build (JIT)"),
                     (r"auto_functionalized was not removed", "crash: torch.compile (Inductor) assertion"),
                     (r"out of memory|OutOfMemory", "crash: out of memory"), (r"timed out", "crash: timed out")):
        if re.search(pat, e):
            return msg
    last = [l for l in e.strip().splitlines() if l.strip()]
    return "crash: " + (last[-1][:100] if last else "unknown")


def main():
    rows = []
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            version, twin = os.path.basename(f)[:-len(".jsonl")].split("__", 1)
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                rows.append({"engine": r.get("engine", "?").split("-")[0], "version": version, "twin": twin,
                             "config": r.get("config"), "engine_version": r.get("engine_version", ""),
                             "requests": r.get("requests", ""), "wrong_requests": r.get("mismatched_requests", ""),
                             "wrong_tokens": r.get("mismatched_tokens", ""), "init_sec": r.get("init_sec", ""),
                             "gen_sec": r.get("gen_sec", ""), "error": short_error(r) if r.get("error") else ""})
    rows.sort(key=lambda r: (r["engine"], [int(x) for x in r["version"].split(".")], r["twin"], r["config"] or ""))
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote docs/data/sweep.csv", len(rows), "rows")
    for r in rows:
        print(f"{r['engine']:6s} {r['version']:7s} {r['twin']:20s} {r['config']:32s} wrong {r['wrong_requests']!s:>4}/{r['requests']!s:<4} {r['error']}")


if __name__ == "__main__":
    main()
