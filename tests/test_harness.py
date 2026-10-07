"""Engine-independent harness helpers: workloads, scoring with failure details, KV layout views and block
ownership (the parts of GPU localization that do not need a GPU)."""
import numpy as np
import pytest
import torch

from probec.compile import compile_probe
from probec.harness import score, shared_workload, workload
from probec.harness.localize import _kv_view, block_owners
from probec.oracle import Probe
from probec.spec import builtin


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("h"))
    compile_probe(builtin("tiny-llama", layers=2), out, classes=512, pools="128,128,256")
    return Probe(out)


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
