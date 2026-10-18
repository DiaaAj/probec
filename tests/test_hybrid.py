"""Hybrid (Gated DeltaNet) probes on CPU through HF transformers: exactness of chunked prefill and recurrent
decode, the echo-skip mechanism, decoys, and detection of recurrent-state faults (docs/HYBRID.md)."""
import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from probec import check  # noqa: E402
from probec.arch import make_config  # noqa: E402
from probec.certify import certify  # noqa: E402
from probec.compile import compile_probe  # noqa: E402
from probec.oracle import Probe  # noqa: E402
from probec.spec import builtin, from_config  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module", params=["tiny-qwen3.5", "tiny-qwen3.5-gva"])
def hybrid_dir(request, tmp_path_factory):
    out = str(tmp_path_factory.mktemp(request.param))
    compile_probe(builtin(request.param, layers=8), out, program="hybrid", classes=512, pools="128,128,256")
    return out


def test_certify(hybrid_dir):
    rep = certify(hybrid_dir)
    assert rep["ok"], rep


def test_zero_false_alarms(hybrid_dir):
    probe = Probe(hybrid_dir)
    rng = np.random.default_rng(0)
    for attn, dtype in (("eager", torch.float32), ("sdpa", torch.bfloat16)):
        model = check.load(hybrid_dir, attn, dtype)
        prompts = [probe.sample_prompt(rng, n) for n in (300, 140)]
        bad, tot = check.batched_generate(model, probe, prompts, 8, sample=dtype == torch.float32)
        assert tot == 16 and bad == 0


def test_bitwise_internal_states(hybrid_dir):
    probe = Probe(hybrid_dir)
    model = check.load(hybrid_dir, "sdpa", torch.bfloat16)
    prm = probe.sample_prompt(np.random.default_rng(1), 200)
    checked, mism, ok = check.bitwise_internal(model, probe, prm, 6)
    assert checked > 0 and mism == 0 and ok


def test_prompt_residuals_stay_in_the_rounding_basin(hybrid_dir):
    """Junk from prompt tokens whose chains break in prefill must stay small: attention payloads read from
    those tokens are divided by their residual's RMS (session 25)."""
    probe = Probe(hybrid_dir)
    model = check.load(hybrid_dir, "sdpa", torch.bfloat16)
    rng = np.random.default_rng(8)
    drift = max(check.residual_health(model, probe, probe.sample_prompt(rng, 1200))[0] for _ in range(2))
    assert drift < 1e-3, drift


def test_echo_writes_redirect_later_hops(hybrid_dir):
    """Decode-time writes matter: some GDN hops read a slot last written by an earlier echo token."""
    probe = Probe(hybrid_dir)
    p = probe.sample_prompt(np.random.default_rng(2), 400)
    exp = probe.expected(p, 32)
    latest, tok, skips = probe._writers(p), p[-1], 0
    for i, e in enumerate(exp):
        for y, kind in zip(probe._walk(tok, latest), probe.types):
            skips += int(kind != "full_attention" and latest[y][1])
        latest[e - probe.C] = (len(p) + i, True)
        tok = e
    assert skips > 0


def test_decoys_keep_the_oracle_and_stay_exact(hybrid_dir):
    probe = Probe(hybrid_dir)
    rng = np.random.default_rng(4)
    prompts = []
    for n in (700, 1000):
        p = probe.sample_prompt(rng, n)
        d = probe.decoy_variant(p, rng, 12)
        assert d != p and probe.expected(d, 12) == probe.expected(p, 12)
        prompts.append(d)
    model = check.load(hybrid_dir, "sdpa", torch.bfloat16)
    assert check.batched_generate(model, probe, prompts, 12, sample=False)[0] == 0


def test_state_faults_detected(hybrid_dir):
    probe = Probe(hybrid_dir)
    model = check.load(hybrid_dir, "sdpa", torch.float32)
    res = check.state_fault_campaign(model, probe, np.random.default_rng(3), 1, 16, 300)
    for kind in ("state_zero", "state_stale", "state_cross_request", "state_wrong_layer"):
        assert res[kind]["detected"] == 1, (kind, res)
    assert res["conv_zero"]["detected"] == 0, res            # v1 reads only the current conv tap


def test_rejected_draft_left_in_the_state_is_detected(hybrid_dir):
    """A rejected draft e(y) for a class y that the next step reads in a GDN layer: if the engine discards
    the draft's KV but not its state update, that hop skips to nxt(nxt(y))."""
    probe = Probe(hybrid_dir)
    model = check.load(hybrid_dir, "sdpa", torch.float32)
    p = probe.sample_prompt(np.random.default_rng(5), 300)
    exp = probe.expected(p, 8)
    latest = probe._writers(p)
    latest[exp[0] - probe.C] = (len(p), True)
    walk = probe._walk(exp[0], latest)
    y = next(y for y, kind in zip(walk, probe.types) if kind != "full_attention" and not latest[y][1])
    tok0, cache = check.prefill(model, p)
    assert tok0 == exp[0]
    model(input_ids=torch.tensor([[probe.C + y]]), past_key_values=cache, use_cache=True,
          attention_mask=torch.ones((1, len(p) + 1), dtype=torch.long), position_ids=torch.tensor([[len(p)]]))
    for layer in cache.layers:
        if hasattr(layer, "keys"):
            layer.keys, layer.values = layer.keys[:, :, :len(p)], layer.values[:, :, :len(p)]
    got = check.decode(model, cache, exp[0], 8, len(p))
    assert got[0] == exp[0] and got[1] != exp[1]


def test_kv_faults_in_the_attention_layers(hybrid_dir):
    probe = Probe(hybrid_dir)
    model = check.load(hybrid_dir, "sdpa", torch.float32)
    res = check.fault_campaign(model, probe, np.random.default_rng(2), 1, 16, 300)
    assert res["zero_block"]["detected"] == 1 and res["zero_block"]["localized"] == 1, res


def test_scale_check_flags_softmax_scale_faults(hybrid_dir):
    probe = Probe(hybrid_dir)
    s_err = probe.meta["scale_check"]["s_err"]
    model = check.load(hybrid_dir, "sdpa", torch.float32)
    attn = [m.self_attn for m in model.model.layers if hasattr(m, "self_attn")]
    base = [a.scaling for a in attn]
    prompt = probe.sample_prompt(np.random.default_rng(1), 200)
    for fac, should_flag in ((1.0, False), (0.5, True), (2.0, True)):
        for a, b in zip(attn, base):
            a.scaling = b * fac
        got = model.generate(input_ids=torch.tensor([prompt]), max_new_tokens=3, min_new_tokens=3,
                             do_sample=False, pad_token_id=probe.pad)[0, len(prompt):].tolist()
        assert (all(t == s_err for t in got) if should_flag else got == probe.expected(prompt, 3)), (fac, got)


def test_qwen3_5_twin_is_its_text_model():
    cfg = json.load(open(os.path.join(DATA, "Qwen_Qwen3.5-0.8B.json")))
    spec = from_config(cfg, layers=8)
    assert spec.arch == "Qwen3_5ForCausalLM" and spec.family == "qwen3_5"
    assert spec.partial_rotary == 0.25 and spec.attn_gate and spec.norm_offset == 1.0
    assert spec.linear["layer_types"].count("full_attention") == 2 and spec.linear["k_heads"] == 16
    out = make_config(spec, spec.rope_theta)
    assert out["architectures"] == ["Qwen3_5ForCausalLM"] and out["model_type"] == "qwen3_5_text"
    assert out["rope_parameters"]["mrope_section"] == [11, 11, 10]
    assert out["rope_parameters"]["partial_rotary_factor"] == 0.25 and out["tie_word_embeddings"] is False
    assert "vision_config" not in out and len(out["layer_types"]) == 8
    assert out["partial_rotary_factor"] == 0.25 and out["full_attention_interval"] == 4   # what SGLang reads


def test_hybrid_matrices():
    from probec.harness import sglang_engine, vllm_engine
    for eng in (vllm_engine, sglang_engine):
        names = [e["name"] for e in eng.matrix("hybrid", "hybrid")]
        assert len(names) == len(set(names)) > 8
        assert any("ngram" in n for n in names) and any("shared_prefix" in n for n in names)
        assert len(eng.matrix("quick", "hybrid")) == 1


def test_moe_hybrid_twin_exact(tmp_path):
    """Qwen3.5-MoE: zero routed experts (any routing gives exactly 0) and the pointer logic in the gated
    shared expert."""
    out = str(tmp_path / "moe")
    compile_probe(builtin("tiny-qwen3.5-moe"), out, program="hybrid", classes=512, pools="128,128,256")
    assert certify(out)["ok"]
    probe = Probe(out)
    model = check.load(out, "sdpa", torch.bfloat16)
    assert type(model.model.layers[0].mlp).__name__ == "Qwen3_5MoeSparseMoeBlock"
    rng = np.random.default_rng(6)
    assert check.batched_generate(model, probe, [probe.sample_prompt(rng, n) for n in (300, 140)], 8,
                                  sample=False) == (0, 16)
    checked, mism, ok = check.bitwise_internal(model, probe, probe.sample_prompt(rng, 150), 4)
    assert checked > 0 and mism == 0 and ok


def test_qwen3_next_twin_exact(tmp_path):
    """Qwen3-Next: the same GDN with fused, per-key-head interleaved in_proj_qkvz / in_proj_ba, and MoE (per-expert
    weights, all zero) with a gated shared expert in every layer."""
    out = str(tmp_path / "next")
    compile_probe(builtin("tiny-qwen3-next"), out, program="hybrid", classes=512, pools="128,128,256")
    assert certify(out)["ok"]
    probe = Probe(out)
    model = check.load(out, "sdpa", torch.bfloat16)
    assert type(model.model.layers[0].linear_attn).__name__ == "Qwen3NextGatedDeltaNet"
    rng = np.random.default_rng(7)
    assert check.batched_generate(model, probe, [probe.sample_prompt(rng, n) for n in (300, 140)], 8,
                                  sample=False) == (0, 16)
    checked, mism, ok = check.bitwise_internal(model, probe, probe.sample_prompt(rng, 150), 4)
    assert checked > 0 and mism == 0 and ok
