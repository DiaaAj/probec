"""Engine harness: run a probe through an inference engine across a matrix of serving configurations,
compare every generated token with the oracle, and optionally inject KV faults to prove detection."""
import collections
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading

import numpy as np

from ..oracle import Probe


def lengths(rng, n, max_len):
    out = []
    for _ in range(n):
        u = rng.random()
        if u < 0.70:
            out.append(int(rng.integers(130, min(1500, max_len))))
        elif u < 0.95:
            out.append(int(rng.integers(min(1500, max_len - 1), min(6000, max_len))))
        else:
            out.append(int(rng.integers(min(6000, max_len - 1), max_len)))
    return out


def shared_workload(probe, rng, n_req, max_len, suffix=(1, 64)):
    """One long base prompt and n_req - 1 variants sharing it (the base runs first to warm the cache).
    A long suffix (e.g. (600, 1200)) makes the engine prefill the new tokens against the cached context."""
    base = probe.sample_prompt(rng, max(2200, (max_len - suffix[1]) // 2))
    return [base] + [probe.shared_variant(base, rng, suffix) for _ in range(n_req - 1)]


def workload(probe, rng, n_req, max_len, shared_frac=0.3):
    """Requests from random pools; a fraction share a long prefix with an earlier request
    (same pool, same body, extra fillers before the final class token) to exercise prefix caching."""
    prompts = []
    for n in lengths(rng, n_req, max_len):
        if prompts and rng.random() < shared_frac:
            prompts.append(probe.shared_variant(prompts[int(rng.integers(0, len(prompts)))], rng))
        else:
            prompts.append(probe.sample_prompt(rng, n))
    return prompts


def decoy_workload(probe, rng, n_req, max_len, n_new, shared_frac=0.3):
    """The default workload with output fragments planted in every fresh prompt (speculative decoding:
    drafts are proposed, accepted and rejected); shared-prefix variants inherit their base's decoys, so
    they still hit the prefix cache. Recall-type probes only."""
    if not hasattr(probe, "decoy_variant"):
        raise SystemExit(f"probec: the decoy workload needs a recall-type probe (got {probe.program})")
    prompts = []
    for n in lengths(rng, n_req, max_len):
        if prompts and rng.random() < shared_frac:
            prompts.append(probe.shared_variant(prompts[int(rng.integers(0, len(prompts)))], rng))
        else:
            prompts.append(probe.decoy_variant(probe.sample_prompt(rng, n), rng, n_new))
    return prompts


def make_workload(entry, probe, rng, n_req, max_len, n_new):
    """Prompts for a matrix entry: default, fresh (no shared prefixes), decoys, shared (short suffix) or
    shared_long (long suffix).
    Returns (prompts, number of leading prompts to run first to warm the prefix cache)."""
    kind = entry.get("workload")
    if kind in ("shared", "shared_long"):
        suffix = (600, 1200) if kind == "shared_long" else (1, 64)
        return shared_workload(probe, rng, n_req, max_len, suffix), 1
    if kind == "decoys":
        return decoy_workload(probe, rng, n_req, max_len, n_new), 0
    if kind == "fresh":                                # no shared-prefix variants: every prompt its own
        return workload(probe, rng, n_req, max_len, shared_frac=0.0), 0
    if kind == "split_markers":                        # markers on both sides of the cached prefix
        base = probe.sample_prompt(rng, max(2200, (max_len - 1200) // 2))
        return [base] + [probe.split_marker_variant(base, rng) for _ in range(n_req - 1)], 1
    return workload(probe, rng, n_req, max_len), 0


def score(probe, prompts, outputs, n_new, diag=None, details=None, max_details=12):
    """outputs: list of generated token-id lists. Returns (bad tokens, bad requests, first bad steps).
    diag: dict counting diagnostic tokens (MoE: misrouted expert ids; recall: 'scale' faults).
    details: list receiving up to max_details failing requests (prompt length, first bad step, context,
    and the prompt positions the first wrong step reads, for localization)."""
    bad_tok = bad_req = 0
    first_bad = []
    for i, (p, got) in enumerate(zip(prompts, outputs)):
        exp = probe.expected(p, n_new)
        d = [j for j, (a, b) in enumerate(zip(got, exp)) if a != b] + ([len(got)] if len(got) != n_new else [])
        bad_tok += len(d)
        bad_req += int(bool(d))
        if not d:
            continue
        j = d[0]
        first_bad.append(j)
        kind = None
        if j < len(got):
            if hasattr(probe, "diagnose") and probe.diagnose(got[j]) is not None:
                kind = f"expert {probe.diagnose(got[j])}"
            elif hasattr(probe, "scale_fault") and probe.scale_fault(got[j]):
                kind = "softmax scale"
        if diag is not None and kind:
            diag[kind] = diag.get(kind, 0) + 1
        if details is not None and len(details) < max_details:
            try:
                reads = probe.reads(p, exp, j) if j < n_new else []
            except Exception:
                reads = []
            details.append({"request": i, "prompt_len": len(p), "first_bad_step": j, "n_bad": len(d),
                            "got": got[max(0, j - 2):j + 3], "expected": exp[max(0, j - 2):j + 3],
                            "diagnosis": kind, "reads_at_first_bad": reads[:16]})
    return bad_tok, bad_req, first_bad


def probe_kind(model_dir):
    with open(os.path.join(model_dir, "probe_meta.json")) as f:
        return json.load(f)["program"]


def _stop_group(proc):
    """Kill a child's whole process group: engine processes a crashed child leaves behind hold GPU memory."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


ISOLATED_ENV = "PROBEC_ISOLATED_ENTRY"      # set in an entry's own process: run it here, do not isolate again


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def run_isolated(names, make_cmd, out, timeout, extra=None):
    """Run each matrix entry in its own child process and session and append its record to `out`.
    make_cmd(name, record_path) gives the child's argv; the child writes its record to record_path. A
    child that exits, is killed or times out without writing one gets an error record with the end of
    its output. Returns the records."""
    # a parent stopped by SIGTERM (e.g. `timeout`) must still stop its child, which has its own session
    trap = threading.current_thread() is threading.main_thread() and \
        signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    if trap:
        signal.signal(signal.SIGTERM, _exit_on_signal)
    recs = []
    try:
        for name in names:
            recs.append(_run_one(name, make_cmd, out, timeout, extra))
    finally:
        if trap:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
    return recs


def _run_one(name, make_cmd, out, timeout, extra):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "record.jsonl")
        proc = subprocess.Popen(make_cmd(name, path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, text=True, errors="replace",
                                env=dict(os.environ, **{ISOLATED_ENV: "1"}))
        tail = collections.deque(maxlen=20)

        def pump():
            for line in proc.stdout or ():
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line.rstrip("\n"))

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            _stop_group(proc)
        reader.join(5)
        rec = None
        if os.path.exists(path):
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            try:
                rec = json.loads(lines[-1]) if lines else None
            except ValueError:          # the child died while writing it
                rec = None
        if rec is None:
            if code is None:
                why = f"timed out after {timeout} s"
            elif code < 0:
                try:
                    why = f"killed by {signal.Signals(-code).name}"
                except ValueError:
                    why = f"killed by signal {-code}"
            else:
                why = f"exited with status {code}"
            rec = {"config": name, **(extra or {}), "error": f"entry process {why} without a record",
                   "output_tail": "\n".join(tail)[-2000:]}
            print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _entry_cmd(a, name, path):
    """argv that runs one matrix entry of this `probec run` in a fresh interpreter."""
    return [sys.executable, "-m", "probec.cli", "run", a.probe, "--engine", a.engine, "--matrix", a.matrix,
            "--only", name, "--requests", str(a.requests), "--new-tokens", str(a.new_tokens),
            "--max-len", str(a.max_len), "--fault-requests", str(getattr(a, "fault_requests", 64)),
            "--out", path]


def run_engine(a):
    if a.engine == "vllm":
        from . import vllm_engine as eng
    else:
        from . import sglang_engine as eng
    kind = {"mla": "mla", "hybrid": "hybrid"}.get(probe_kind(a.probe), "dense")
    plan = eng.matrix(a.matrix, kind)
    if a.list:
        print("\n".join(e["name"] for e in plan))
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if not getattr(a, "in_process", False) and not os.environ.get(ISOLATED_ENV):
        # SGLang kills its whole process tree when a scheduler fails (e.g. a backend this GPU cannot
        # compile), which in one process would end the run at the first such entry; and an engine torn
        # down in-process can leave GPU memory allocated, so the next one would not fit (seen with vLLM)
        names = [e["name"] for e in plan if not a.only or e["name"] == a.only]
        recs = run_isolated(names, lambda name, path: _entry_cmd(a, name, path), a.out,
                            getattr(a, "entry_timeout", 1800), extra={"engine": a.engine})
        total = sum(r.get("gen_tokens", 0) for r in recs)
        bad = sum(r.get("mismatched_tokens", 0) or 0 for r in recs)
        print(f"probec run: {total} generated tokens checked, {bad} mismatched")
        return 2 if (bad and getattr(a, "strict", False)) else 0
    probe = Probe(a.probe)
    max_len = min(a.max_len, probe.max_context - a.new_tokens - 64)
    total = bad = 0
    for i, entry in enumerate(plan):
        if a.only and entry["name"] != a.only:
            continue
        rng = np.random.default_rng(entry.get("seed", i))
        try:
            rec = eng.run_entry(entry, probe, a, rng, max_len)
        except Exception as e:          # keep going: a backend may be unavailable on this GPU
            rec = {"config": entry["name"], "engine": a.engine, "error": repr(e)[:1000]}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        total += rec.get("gen_tokens", 0)
        bad += rec.get("mismatched_tokens", 0)
    print(f"probec run: {total} generated tokens checked, {bad} mismatched")
    return 2 if (bad and getattr(a, "strict", False)) else 0
