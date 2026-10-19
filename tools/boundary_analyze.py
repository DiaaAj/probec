"""Summarize tools/fault_boundary.py records into docs/data/boundary_faults.csv (per fault and size: trials,
landed, read, detected, localized, steps from the first read to the first wrong token),
docs/data/boundary_scale.csv and docs/data/boundary_coverage.csv.

With --twin (the twin fault_boundary.py ran, e.g. `probec compile --like tests/data/Qwen_Qwen3-0.6B.json
--no-scale-check`), it also replays the oracle on the trial prompts: for every detected content fault, the
step at which the oracle predicts the first read of a corrupted (layer, position) versus the engine's first
wrong step (docs/data/readmap_agreement.csv), and the reads of one single-slot, single-layer fault for the
report's figure (docs/data/readmap_reads.csv, readmap_trial.json).

    python tools/boundary_analyze.py out/boundary.jsonl [--twin probes/<the twin fault_boundary.py ran>]
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


def replay(recs, twin):
    """Oracle replay of the kv trials: predicted first read of the corrupted data vs the engine's first wrong step."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import numpy as np
    from probec.harness import workload
    from probec.oracle import Probe
    hdr = next(r for r in recs if r["part"] == "kv_header")
    probe = Probe(twin)
    rng = np.random.default_rng(5)
    n = hdr["prompts"]
    prompts = [p for p in workload(probe, rng, 4 * n, 4000) if 600 <= len(p) <= 4000][:n]
    assert [len(p) for p in prompts] == hdr["prompt_lens"], "not the GPU job's prompts (twin or seed differs)"
    bs, L, n_new = hdr["block_size"], hdr["layers"], hdr["new_tokens"]
    cache = {}

    def reads(r):                                       # [(step, layer, position)] of request r's generation
        if r not in cache:
            p = prompts[r]
            exp = probe.expected(p, n_new)
            where = {t: i for i, t in enumerate(p) if t < probe.C}
            out = []
            for s in range(n_new):
                prev = p[-1] if s == 0 else exp[s - 1]
                first = 0 if prev < probe.C else 1      # after an echo token, layer 0 reads the echo itself
                out += [(s, first + k, where[c]) for k, c in enumerate(probe.chain(prev)) if c in where]
            cache[r] = out
        return cache[r]

    agree, pick = [], None
    content = {"block_same_request", "block_other_request", "block_zeroed", "block_layer_below", "block_stale_one_layer",
               "one_slot", "one_slot_one_layer", "one_kv_head", "negate_k", "negate_v", "block_other_request_random"}
    for t in recs:
        if t["part"] != "kv" or t["fault"] not in content:
            continue
        layers = set(t.get("layers") or range(1 if t["fault"] == "block_layer_below" else 0, L))
        hit = [s for s, l, pos in reads(t["request"]) if l in layers and
               (pos == t["slot_pos"] if t.get("slot_pos") is not None else pos // bs == t["block"])]
        pred = min(hit) if hit else None
        agree.append({"fault": t["fault"], "request": t["request"], "block": t["block"], "slot_pos": t.get("slot_pos"),
                      "predicted_step": pred, "detected": int(t["detected"]), "first_bad_step": t["first_bad_step"],
                      "observed_minus_predicted": "" if pred is None or not t["detected"] else t["first_bad_step"] - pred})
        if t["fault"] == "one_slot_one_layer" and t["detected"] and t.get("slot_pos") is not None and pred is not None and \
                (pick is None or abs(t["first_bad_step"] - 30) < abs(pick["first_bad_step"] - 30)):
            pick = t
    write("docs/data/readmap_agreement.csv", agree)
    d = [a["observed_minus_predicted"] for a in agree if a["observed_minus_predicted"] != ""]
    print(f"oracle replay: {sum(1 for x in d if x == 0)} of {len(d)} detected content faults went wrong exactly at the "
          f"predicted step; others: {sorted(x for x in d if x != 0)}")
    both = collections.Counter((a["predicted_step"] is not None, bool(a["detected"])) for a in agree)
    print(f"content faults: read and detected {both[(True, True)]}, read but missed {both[(True, False)]}, "
          f"never read and missed {both[(False, False)]}, never read but detected {both[(False, True)]}")
    if pick:
        lay = pick["layers"][0]
        write("docs/data/readmap_reads.csv", [{"step": s, "layer": l} for s, l, pos in reads(pick["request"])
                                              if pos == pick["slot_pos"]])
        with open("docs/data/readmap_trial.json", "w") as f:
            json.dump({"request": pick["request"], "position": pick["slot_pos"], "layer": lay, "layers": L,
                       "first_bad_step": pick["first_bad_step"], "fault": pick["fault"],
                       "engine_version": hdr["engine_version"]}, f, indent=1)
        print("readmap trial:", pick["fault"], "request", pick["request"], "position", pick["slot_pos"], "layer", lay,
              "first wrong step", pick["first_bad_step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--twin")
    a = ap.parse_args()
    recs = [json.loads(l) for p in a.records for l in open(p) if l.strip()]
    kv = collections.OrderedDict()
    scale, cov = [], []
    for r in recs:
        if r["part"] == "kv":
            kv.setdefault((r["fault"], r.get("eps")), []).append(r)
        elif r["part"] == "scale":
            scale.append({k: r[k] for k in ("factor", "requests", "wrong_requests", "wrong_tokens", "scale_diagnosed",
                                            "median_first_bad_step", "layers_patched", "engine_version")})
        elif r["part"] == "kv_header":
            cov += [{"request": i, "prompt_len": n, "read_block_fraction": f}
                    for i, (n, f) in enumerate(zip(r["prompt_lens"], r["read_block_fraction"]))]
            print("header:", {k: v for k, v in r.items() if k not in ("prompt_lens", "read_block_fraction")})
        elif r["part"] == "kv_baseline":
            print("baseline:", r)
    rows = []
    for (fault, eps), rs in kv.items():
        read = [r for r in rs if r["read_step"] is not None]
        det = [r for r in rs if r["detected"]]
        lat = [r["first_bad_step"] - r["read_step"] for r in det if r["read_step"] is not None]
        rows.append({"fault": fault, "eps": "" if eps is None else eps, "trials": len(rs),
                     "landed": sum(1 for r in rs if r["changed"] != 0), "read": len(read), "detected": len(det),
                     "detected_when_read": sum(1 for r in read if r["detected"]),
                     "detected_when_unread": sum(1 for r in rs if r["read_step"] is None and r["detected"]),
                     "localized": sum(1 for r in det if r.get("localized")),
                     "median_steps_read_to_wrong": statistics.median(lat) if lat else "",
                     "median_cache_hit_tokens": statistics.median(r["cache_hit_tokens"] for r in rs)})
    os.makedirs("docs/data", exist_ok=True)
    write("docs/data/boundary_faults.csv", rows)
    if scale:
        write("docs/data/boundary_scale.csv", scale)
    if cov:
        write("docs/data/boundary_coverage.csv", cov)
    for r in rows:
        print(f"{r['fault']:28s} {str(r['eps']):6s} detected {r['detected']:2d}/{r['trials']:<2d} landed {r['landed']:2d} "
              f"read {r['read']:2d} det|read {r['detected_when_read']:2d} det|unread {r['detected_when_unread']} "
              f"localized {r['localized']:2d} latency {r['median_steps_read_to_wrong']}")
    for s in scale:
        print(f"scale x{s['factor']:<8} wrong {s['wrong_requests']}/{s['requests']} diagnosed {s['scale_diagnosed']}")
    if a.twin:
        replay(recs, a.twin)


if __name__ == "__main__":
    main()
