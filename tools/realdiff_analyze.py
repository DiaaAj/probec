"""Compare tools/realdiff.py records: how often correct configurations of the same engine disagree on a real
model, whether the configurations the probe flags stand out, and whether a logprob-tolerance check (the
kind engine test suites use) passes. Writes docs/data/realdiff_pairs.csv and docs/data/realdiff_configs.csv.

    python tools/realdiff_analyze.py out/realdiff [more directories ...]
"""
import collections
import csv
import glob
import gzip
import json
import os
import sys

REFERENCE = {"vllm": "FLASH_ATTN/default", "sglang": "flashinfer/default"}


def load(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
            with gzip.open(f, "rt") as fh:
                recs.append(json.load(fh))
    return recs


def label(r):
    return f"{r['engine']}:{r['config']}" + ("" if r.get("logprobs", 1) else " (no logprobs)")


def norm(ids, stops):
    while ids and ids[-1] in stops:
        ids = ids[:-1]
    return ids


def logprob_close(a, b):
    """vLLM's check_logprobs_close for one request: identical, or at the first differing step each side's
    token is among the other side's top-k. Compares the common length only, as that check does."""
    ia, ib = a["ids"], b["ids"]
    for i, (x, y) in enumerate(zip(ia, ib)):
        if x == y:
            continue
        ta, tb = (a.get("top") or [None] * len(ia))[i], (b.get("top") or [None] * len(ib))[i]
        if ta is None or tb is None:
            return None                                    # no logprobs recorded
        return x in {t for t, _ in tb} and y in {t for t, _ in ta}
    return True


def compare(a, b, stops):
    n = differ = lp_fail = lp_na = 0
    first = []
    for oa, ob in zip(a["outputs"], b["outputs"]):
        n += 1
        xa, xb = norm(oa["ids"], stops), norm(ob["ids"], stops)
        if xa != xb:
            differ += 1
            first.append(next((i for i, (x, y) in enumerate(zip(xa, xb)) if x != y), min(len(xa), len(xb))))
        ok = logprob_close(oa, ob)
        if ok is None:
            lp_na += 1
        elif not ok:
            lp_fail += 1
    return {"requests": n, "differ": differ, "differ_frac": round(differ / max(1, n), 4),
            "median_first_diff": sorted(first)[len(first) // 2] if first else "",
            "logprob_check_failed": lp_fail if lp_na < n else "", "logprob_na": lp_na}


def main():
    recs = load(sys.argv[1:] or ["out/realdiff"])
    real = [r for r in recs if r["side"].startswith("real")]
    probe = [r for r in recs if r["side"].startswith("probe")]
    # a setting is flagged when the same run (tag) got the probe twin wrong under it
    flagged = {(r["tag"], r["config"]) for r in probe if r.get("mismatched_requests")}
    by_model = collections.defaultdict(list)
    for r in real:
        by_model[r["model"]].append(r)
    pairs, configs = [], []
    for model, rs in sorted(by_model.items()):
        digests = {r["prompt_digest"] for r in rs}
        if len(digests) != 1:
            print(f"WARNING {model}: different prompts across records: {sorted(digests)}")
        stops = set(rs[0]["stops"])
        rs.sort(key=label)
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                c = compare(a, b, stops)
                pairs.append({"model": model, "a": label(a), "b": label(b), **c,
                              "a_flagged": int((a["tag"], a["config"]) in flagged),
                              "b_flagged": int((b["tag"], b["config"]) in flagged)})
        for r in rs:
            ref = next((x for x in rs if x["engine"] == r["engine"] and x["config"] == REFERENCE.get(r["engine"])
                        and x.get("logprobs") == r.get("logprobs")), None)
            c = compare(r, ref, stops) if ref is not None and ref is not r else {}
            sd = r.get("spec_decode") or {}
            configs.append({"model": model, "side": r["side"], "engine": r["engine"], "engine_version": r.get("engine_version"),
                            "config": r["config"], "logprobs": r.get("logprobs"), "requests": r["requests"],
                            "reference": REFERENCE.get(r["engine"], "") if c else "",
                            "differ_vs_reference": c.get("differ", ""), "differ_frac_vs_reference": c.get("differ_frac", ""),
                            "logprob_check_failed_vs_reference": c.get("logprob_check_failed", ""),
                            "probe_flags_config": int((r["tag"], r["config"]) in flagged),
                            "spec_decode": json.dumps(sd) if sd else ""})
    for r in sorted(probe, key=lambda r: (r["tag"], r["config"], r["side"])):
        configs.append({"model": r.get("twin"), "side": r["side"], "engine": r["engine"], "engine_version": r.get("engine_version"),
                        "config": r["config"], "logprobs": r.get("logprobs"), "requests": r["requests"],
                        "wrong_requests": r.get("mismatched_requests"), "wrong_tokens": r.get("mismatched_tokens"),
                        "probe_flags_config": int(bool(r.get("mismatched_requests"))),
                        "spec_decode": json.dumps(r.get("spec_decode") or {}) if r.get("spec_decode") else ""})
    os.makedirs("docs/data", exist_ok=True)
    for path, rows in (("docs/data/realdiff_pairs.csv", pairs), ("docs/data/realdiff_configs.csv", configs)):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, len(rows), "rows")
    for c in configs:
        if c["side"].startswith("real"):
            print(f"{c['model']:20s} {c['engine']:6s} {c['config']:28s} lp={c['logprobs']} differ_vs_ref={c['differ_vs_reference']!s:>4} "
                  f"lp_fail={c['logprob_check_failed_vs_reference']!s:>4} flagged_by_probe={c['probe_flags_config']}")
        else:
            print(f"{c['model']:20s} {c['engine']:6s} {c['config']:28s} lp={c['logprobs']} PROBE wrong {c['wrong_requests']}/{c['requests']}")


if __name__ == "__main__":
    main()
