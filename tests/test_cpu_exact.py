"""End-to-end CPU tests: compile small probes, check zero false alarms, bit-exact internal states, and that
injected faults are detected (and localized / named). Uses HF transformers as the engine."""
import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from probec import check  # noqa: E402
from probec.certify import certify  # noqa: E402
from probec.compile import compile_probe  # noqa: E402
from probec.oracle import Probe  # noqa: E402
from probec.spec import builtin  # noqa: E402

CASES = [  # (builtin shape, program, compile kwargs)
    ("tiny-llama", "recall", dict(classes=512, pools="128,128,256")),
    ("tiny-qwen3", "recall", dict(classes=512, pools="128,128,256")),
    ("tiny-qwen3-moe", "moe", dict(classes=512, pools="128,128,256")),
    ("tiny-deepseek-v3", "mla", dict(classes=512, pools="128,128,256")),
    ("tiny-deepseek-v32", "mla", dict(classes=512, pools="128,128,256")),
    ("tiny-llama", "posrecall", dict(classes=512)),
]


@pytest.fixture(scope="module", params=CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def probe_dir(request, tmp_path_factory):
    shape, program, kw = request.param
    out = str(tmp_path_factory.mktemp(f"{shape}-{program}"))
    compile_probe(builtin(shape, layers=4), out, program=program, **kw)
    return out


def _lengths(probe):
    return (140, 70) if probe.program != "posrecall" else (probe.min_len + 20, probe.min_len + 90)


def test_certify(probe_dir):
    rep = certify(probe_dir)
    assert rep["ok"], rep


def test_zero_false_alarms(probe_dir):
    probe = Probe(probe_dir)
    rng = np.random.default_rng(0)
    runs = [("eager", torch.float32), ("sdpa", torch.bfloat16)] if probe.program != "posrecall" \
        else [("sdpa", torch.float32)]
    for attn, dtype in runs:
        model = check.load(probe_dir, attn, dtype)
        prompts = [probe.sample_prompt(rng, n) for n in _lengths(probe)]
        bad, tot = check.batched_generate(model, probe, prompts, 8, sample=dtype == torch.float32)
        assert tot == 16 and bad == 0


def test_bitwise_internal_states(probe_dir):
    # bf16 is the serving dtype: rounding basins absorb the small perturbation that the positional probe's
    # undefined reads (offset > position, early tokens only) leave in fp32 internal states
    probe = Probe(probe_dir)
    model = check.load(probe_dir, "sdpa", torch.bfloat16)
    prm = probe.sample_prompt(np.random.default_rng(1), _lengths(probe)[0])
    checked, mism, ok = check.bitwise_internal(model, probe, prm, 4)
    assert checked > 0 and mism == 0 and ok


def test_faults_detected(probe_dir):
    probe = Probe(probe_dir)
    model = check.load(probe_dir, "sdpa", torch.float32)
    rng = np.random.default_rng(2)
    if probe.program == "posrecall":
        res = check.position_faults(model, probe, rng, 1, 12, probe.min_len + 100)
        for kind in ("decode_position_shift", "rope_theta_drift"):
            assert res[kind]["detected"] == 1, (kind, res)
        return
    res = check.fault_campaign(model, probe, rng, 1, 16, 300)
    for kind in ("block_dup", "cross_request", "zero_block", "wrong_layer"):
        assert res[kind]["detected"] == 1 and res[kind]["localized"] == 1, (kind, res)
    if probe.program == "moe":
        rr = check.routing_faults(model, probe, rng, 2, 16, 300)
        for kind in ("misroute", "duplicate", "drop"):
            assert rr[kind]["detected"] == 2, (kind, rr)
        assert rr["misroute"]["named"] == 2 and rr["duplicate"]["named"] == 2, rr


@pytest.mark.parametrize("shape,program", [("tiny-llama", "recall"), ("tiny-qwen3", "recall"),
                                           ("tiny-deepseek-v3", "mla"), ("tiny-deepseek-v32", "mla")])
def test_scale_check_flags_softmax_scale_faults(shape, program, tmp_path):
    out = str(tmp_path / shape)
    compile_probe(builtin(shape, layers=3), out, program=program, classes=512, pools="128,128,256")
    probe = Probe(out)
    s_err = probe.meta["scale_check"]["s_err"]
    model = check.load(out, "sdpa", torch.float32)
    base = [m.self_attn.scaling for m in model.model.layers]
    prompt = probe.sample_prompt(np.random.default_rng(1), 200)
    for fac, should_flag in ((1.0, False), (1.1, False), (0.5, True), (1.4427, True), (2.0, True), (11.3137, True)):
        for m, b in zip(model.model.layers, base):
            m.self_attn.scaling = b * fac
        got = model.generate(input_ids=torch.tensor([prompt]), max_new_tokens=4, min_new_tokens=4,
                             do_sample=False, pad_token_id=probe.pad)[0, len(prompt):].tolist()
        if should_flag:
            assert all(t == s_err for t in got), (fac, got)
        else:
            assert got == probe.expected(prompt, 4), (fac, got)


@pytest.mark.parametrize("shape,program", [("tiny-qwen3", "recall"), ("tiny-deepseek-v3", "mla")])
def test_decoy_prompts_exact(shape, program, tmp_path):
    """Output fragments planted in the prompt (speculative-decoding workload) leave the output exact."""
    out = str(tmp_path / shape)
    compile_probe(builtin(shape, layers=2), out, program=program, classes=512, pools="128,128,256")
    probe = Probe(out)
    model = check.load(out, "sdpa", torch.bfloat16)
    rng = np.random.default_rng(4)
    prompts = [probe.decoy_variant(probe.sample_prompt(rng, n), rng, 12) for n in (800, 1100)]
    assert check.batched_generate(model, probe, prompts, 12, sample=False)[0] == 0


def _fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)


def test_fp8_attention_paths_exact(tmp_path):
    """Q, K and V rounded to FP8 e4m3 (what FP8 KV caches and FA3 FP8 do) leave dense outputs exact,
    scale check included."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def fp8_attention(module, q, k, v, mask, **kw):
        return sdpa_attention_forward(module, _fp8(q), _fp8(k), _fp8(v), mask, **kw)
    AttentionInterface.register("probec_fp8", fp8_attention)
    out = str(tmp_path / "fp8")
    compile_probe(builtin("tiny-qwen3", layers=2), out, classes=512, pools="128,128,256")
    probe = Probe(out)
    model = check.load(out, "probec_fp8", torch.bfloat16)
    rng = np.random.default_rng(6)
    prompts = [probe.sample_prompt(rng, n) for n in (300, 900)]
    assert check.batched_generate(model, probe, prompts, 8, sample=False)[0] == 0


def test_mla_scale_check_survives_fp8_absorbed_query(tmp_path):
    """FP8 MLA kernels quantize the absorbed query W_UK^T q (and P); the calibrated head must stay inside
    its window, with a static scale of 1 or a dynamic per-token scale."""
    import glob
    from safetensors import safe_open
    out = str(tmp_path / "mla")
    compile_probe(builtin("tiny-deepseek-v3", layers=2), out, program="mla", classes=512, pools="128,128,256")
    meta = Probe(out).meta
    cfg = json.load(open(os.path.join(out, "config.json")))
    nope, rope, vh, kvl, ql = (cfg[k] for k in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                                                "kv_lora_rank", "q_lora_rank"))
    nh, pre = cfg["num_attention_heads"], f"model.layers.{cfg['num_hidden_layers'] - 1}.self_attn."
    w = {}
    for fn in glob.glob(os.path.join(out, "*.safetensors")):
        with safe_open(fn, "pt") as f:
            w.update({k: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)})
    qlat = torch.ones(ql)
    q = (w[pre + "q_b_proj.weight"] @ qlat).view(nh, nope + rope)[:, :nope]
    qabs = torch.einsum("hn,hnc->hc", q, w[pre + "kv_b_proj.weight"].view(nh, nope + vh, kvl)[:, :nope])
    lat = {}
    for kind, (m, ab) in {"A": (1, 1), "B": (1, -1), "N": (-1, 1)}.items():
        v = torch.ones(kvl)
        v[128], v[129] = m, ab                         # LM, LAB
        lat[kind] = v
    scale, (lo, hi) = meta["softmax_scale"], meta["scale_check"]["window"]
    for qh in (qabs[1], _fp8(qabs[1]), _fp8(qabs[1] / (qabs.abs().max() / 448)) * (qabs.abs().max() / 448)):
        s = torch.stack([qh @ lat[k] for k in "ABN"]) * scale
        p = _fp8(torch.exp(s - s.max()))
        cal = float((p * torch.tensor([1.0, -1.0, 1.0])).sum() / p.sum())
        assert lo < cal < hi, cal


def test_dsa_query_token_drop_detected(tmp_path):
    out = str(tmp_path / "dsa")
    compile_probe(builtin("tiny-deepseek-v32", layers=3), out, program="mla", classes=512, pools="128,128,256")
    probe = Probe(out)
    model = check.load(out, "sdpa", torch.float32)
    res = check.dsa_query_drop(model, probe, np.random.default_rng(0), 2, 8, 300)
    assert res["detected"] == 2, res
