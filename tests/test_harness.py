"""Engine-independent harness helpers: workloads, scoring with failure details, KV layout views, block
ownership (the parts of GPU localization that do not need a GPU) and running entries in their own
processes."""
import json
import sys

import numpy as np
import pytest
import torch

from probec.compile import compile_probe
from probec.harness import run_isolated, score, shared_workload, workload
from probec.harness.localize import _kv_view, block_owners
from probec.oracle import Probe
from probec.spec import builtin


@pytest.fixture(scope="module")
def probe_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("h"))
    compile_probe(builtin("tiny-llama", layers=2), out, classes=512, pools="128,128,256")
    return out


@pytest.fixture(scope="module")
def probe(probe_dir):
    return Probe(probe_dir)


def test_workload_prompts_are_valid(probe):
    ps = workload(probe, np.random.default_rng(0), 20, 3000)
    assert len(ps) == 20 and all(len(p) >= 130 for p in ps)
    for p in ps:
        assert len(probe.expected(p, 4)) == 4


def test_shared_workload_shares_the_base(probe):
    ps = shared_workload(probe, np.random.default_rng(1), 5, 4000, (600, 1200))
    base = ps[0]
    for p in ps[1:]:
        assert p[:len(base) - 1] == base[:-1] and p[-1] == base[-1] and 600 <= len(p) - len(base) < 1200


def test_score_reports_failures_and_scale_diagnosis(probe):
    rng = np.random.default_rng(2)
    prompts = [probe.sample_prompt(rng, 200) for _ in range(3)]
    outs = [probe.expected(p, 8) for p in prompts]
    s_err = probe.meta["scale_check"]["s_err"]
    outs[2] = outs[2][:5] + [s_err] * 3
    diag, details = {}, []
    bad_tok, bad_req, first = score(probe, prompts, outs, 8, diag, details)
    assert (bad_tok, bad_req, first) == (3, 1, [5])
    assert diag == {"softmax scale": 1}
    assert details[0]["request"] == 2 and details[0]["first_bad_step"] == 5


@pytest.mark.parametrize("shape", [(2, 30, 16, 8, 128), (30, 2, 16, 8, 128), (30, 8, 16, 256)])
def test_kv_views_write_through(shape):
    x = torch.randn(*shape)
    v = _kv_view(x, 30, 16, 8, 128)
    assert tuple(v.shape) == (30, 2, 16, 8, 128)
    v[7].zero_()
    assert torch.count_nonzero(_kv_view(x, 30, 16, 8, 128)[7]) == 0


def test_block_owners_lists_every_copy():
    prompts = [list(range(100, 140)), list(range(100, 140)) + [5]]
    slot = [[-1] * 16 for _ in range(8)]
    slot[2] = list(range(100, 116))
    slot[5] = list(range(100, 116))
    slot[4] = list(range(116, 132))
    assert block_owners(slot, prompts, 16)[(0, 0)] == 2
    assert block_owners(slot, prompts, 16, all_copies=True)[(1, 0)] == [2, 5]


def test_decoy_variant_keeps_the_oracle(probe):
    rng = np.random.default_rng(3)
    p = probe.sample_prompt(rng, 1500)
    d = probe.decoy_variant(p, rng, 64)
    assert len(d) == len(p) and d[-1] == p[-1] and d != p
    assert probe.expected(d, 64) == probe.expected(p, 64)
    assert [i for i, t in enumerate(d) if t < probe.C] == [i for i, t in enumerate(p) if t < probe.C]
    assert sum(probe.C <= t < 2 * probe.C for t in d) >= 16 * 6      # 16 fragments of >= 6 echo tokens


def test_spec_stats_parsing():
    from types import SimpleNamespace as NS
    from probec.harness import sglang_engine, vllm_engine
    llm = NS(get_metrics=lambda: [NS(name="vllm:spec_decode_num_drafts", value=10),
                                  NS(name="vllm:spec_decode_num_accepted_tokens_per_pos", values=[5, 3]),
                                  NS(name="vllm:num_requests_running", value=1)])
    assert vllm_engine.spec_stats(llm) == {"num_drafts": 10, "num_accepted_tokens_per_pos": 8}
    st = sglang_engine.spec_stats([{"completion_tokens": 256, "spec_verify_ct": 128, "spec_accept_rate": 0.5},
                                   {"completion_tokens": 256, "spec_verify_ct": 128, "spec_accept_rate": 0.7}])
    assert st == {"spec_verify_ct": 256, "spec_accept_rate": 0.6, "tokens_per_verify_step": 2.0}


def test_decoy_workload_keeps_shared_prefixes(probe):
    from probec.harness import decoy_workload
    ps = decoy_workload(probe, np.random.default_rng(5), 40, 3000, 32)
    planted = [p for p in ps if any(probe.C <= t < 2 * probe.C for t in p)]
    shared = [q for q in ps for p in ps if q is not p and len(q) > len(p) and q[:len(p) - 1] == p[:-1]]
    assert len(planted) > 30 and shared


def test_split_marker_variant_keeps_the_oracle(probe):
    rng = np.random.default_rng(7)
    base = probe.sample_prompt(rng, 2200)
    v = probe.split_marker_variant(base, rng)
    markers = probe.meta["scale_check"]["markers"]
    pos = sorted(i for i, t in enumerate(v) if t in markers)
    first_diff = next(i for i, (a, b) in enumerate(zip(base, v)) if a != b)
    assert len(pos) == 2 and pos[0] < first_diff <= pos[1]       # one marker cached, one in the new part
    assert probe.expected(v, 16) == probe.expected(base, 16)


def test_isolated_entries_survive_crashes_and_timeouts(tmp_path):
    # what SGLang's SIGQUIT handler does when a scheduler fails: kill the whole process tree
    children = {
        "writes": "import json, sys; open(sys.argv[1], 'w').write(json.dumps({'config': 'writes', 'gen_tokens': 8}))",
        "exits": "import sys; print('boom'); sys.exit(3)",
        "kills_its_tree": ("import os, signal, subprocess, sys; "
                           "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                           "os.killpg(0, signal.SIGKILL)"),
        "hangs": "import time; time.sleep(60)",
    }
    out = tmp_path / "results.jsonl"
    recs = run_isolated(list(children), lambda name, path: [sys.executable, "-c", children[name], path],
                        str(out), timeout=3, extra={"engine": "fake"})
    assert recs[0] == {"config": "writes", "gen_tokens": 8}
    assert recs[1]["error"] == "entry process exited with status 3 without a record"
    assert "boom" in recs[1]["output_tail"]
    assert recs[2]["error"] == "entry process killed by SIGKILL without a record"
    assert recs[3]["error"] == "entry process timed out after 3 s without a record"
    assert [r["config"] for r in recs] == list(children) and all(r["engine"] == "fake" for r in recs[1:])
    assert [json.loads(l) for l in out.read_text().splitlines()] == recs


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_each_matrix_entry_runs_in_its_own_process(probe_dir, tmp_path, monkeypatch, engine):
    from types import SimpleNamespace as NS
    import probec.harness as harness
    from probec.harness import sglang_engine, vllm_engine
    seen = {}

    def fake_run_isolated(names, make_cmd, out, timeout, extra=None):
        seen.update(names=names, cmd=make_cmd(names[1], "rec.jsonl"), timeout=timeout, extra=extra)
        return [{"config": n, "gen_tokens": 16, "mismatched_tokens": 0} for n in names]

    monkeypatch.setattr(harness, "run_isolated", fake_run_isolated)
    a = NS(probe=probe_dir, engine=engine, matrix="default", only="", list=False, requests=8, new_tokens=16,
           max_len=4000, fault_requests=64, out=str(tmp_path / "r.jsonl"), strict=True, in_process=False,
           entry_timeout=900)
    assert harness.run_engine(a) == 0
    eng = sglang_engine if engine == "sglang" else vllm_engine
    assert seen["names"] == [e["name"] for e in eng.matrix("default", "dense")]
    cmd = seen["cmd"]
    assert cmd[cmd.index("--only") + 1] == seen["names"][1] and cmd[cmd.index("--out") + 1] == "rec.jsonl"
    assert cmd[cmd.index("--engine") + 1] == engine
    assert seen["timeout"] == 900 and seen["extra"] == {"engine": engine}
