"""Quantized formats: the weight quantizers reproduce designed weights bit for bit, activation fake
quantization keeps on-grid values, and a probe twin stays exact under quantized weights and activations."""
import pytest
import torch

from probec.compile import compile_probe
from probec.quant import (ACT_FORMATS, WEIGHT_FORMATS, certify_weights, check_quant, dequantize,
                          fake_quant_act, quantize_weight)
from probec.spec import builtin


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("q"))
    compile_probe(builtin("tiny-llama", layers=2), out, classes=512, pools="128,128,256", scale_check=False)
    return out


def sparse_weight():
    w = torch.zeros(256, 256)
    w[0, 3], w[1, 5], w[2, 130], w[3, 7], w[3, 9] = 8.0, -0.5, 32.0, 1.0, -2.0
    return w


@pytest.mark.parametrize("fmt", list(WEIGHT_FORMATS))
def test_designed_weights_quantize_exactly(fmt):
    w = sparse_weight()
    assert quantize_weight(w, fmt)[-1]
    assert torch.equal(dequantize(w, fmt), w)


def test_mixed_magnitudes_only_fit_fp8():
    w = sparse_weight()
    w[3, 11] = 0.0625                     # 16x below the group's 1.0: beyond 4-bit and E2M1 ranges
    assert quantize_weight(w, "fp8")[-1] and quantize_weight(w, "fp8-block")[-1]
    for fmt in ("int4-awq", "int4-gptq", "nvfp4", "mxfp4"):
        assert not quantize_weight(w, fmt)[-1]


def test_activation_fake_quant_keeps_grid_values():
    mixed = torch.tensor([[0.125, 2.0, -0.125, 0.0] * 32])            # RMSNorm outputs: codes and ballast
    for fmt in ("fp8-tensor", "fp8-token", "fp8-group128"):
        assert torch.equal(fake_quant_act(mixed, fmt), mixed)
    homogeneous = torch.cat([torch.full((1, 64), 2.0), torch.tensor([[0.125, -0.125] * 32])], dim=1)
    for fmt in ACT_FORMATS:
        assert torch.equal(fake_quant_act(homogeneous, fmt), homogeneous), fmt
    for fmt in ("nvfp4", "mxfp4"):                # 1/8 next to 2 in one FP4 group does not survive
        assert not torch.equal(fake_quant_act(mixed, fmt), mixed)


def test_twin_weights_survive_every_format(twin):
    from probec.certify import _tensors
    rep = certify_weights(_tensors(twin), list(WEIGHT_FORMATS))
    for fmt, r in rep.items():
        assert r["tensors"] > 0 and r["exact"] == r["tensors"], (fmt, r["inexact"])


@pytest.mark.parametrize("act", ["fp8-token", "fp8-group128", "nvfp4"])
def test_twin_stays_exact_under_quantized_activations(twin, act):
    r = check_quant(twin, act=act, weight="fp8", n_prompts=3, max_len=600, n_new=8)
    assert r["tokens"] == 24 and r["mismatches"] == 0, r


def _decode(f, name, w, fmt):
    """The weight an engine reconstructs, by the formulas of vLLM's and SGLang's loaders and kernels."""
    from probec.quant import E2M1, _blocks
    base = name[: -len(".weight")]
    n, k = w.shape
    if fmt == "fp8":
        return f.get_tensor(name).float() * f.get_tensor(base + ".weight_scale")
    if fmt == "fp8-block":
        s = f.get_tensor(base + ".weight_scale_inv")
        t = _blocks(f.get_tensor(name).float(), 128, 128) * s[:, None, :, None]
        return t.reshape(s.shape[0] * 128, -1)[:n, :k]
    if fmt == "int4-awq":
        rev = [0, 4, 1, 5, 2, 6, 3, 7]          # column c of an 8-group sits at shift 4 * rev[c]
        def unpack(v):
            v = v.to(torch.int64) & 0xFFFFFFFF
            return torch.stack([(v >> (4 * rev[c])) & 0xF for c in range(8)], -1).reshape(v.shape[0], -1)
        q, z = unpack(f.get_tensor(base + ".qweight")), unpack(f.get_tensor(base + ".qzeros"))
        s = f.get_tensor(base + ".scales").float()
        g = torch.arange(k) // 128
        return ((q - z[g]).float() * s[g]).t()
    if fmt == "int4-gptq":
        v = f.get_tensor(base + ".qweight").to(torch.int64) & 0xFFFFFFFF
        q = torch.stack([(v >> (4 * j)) & 0xF for j in range(8)], 1).reshape(k, n)    # row 8i+j <- nibble j
        assert bool((f.get_tensor(base + ".qzeros") == 0x77777777).all())
        assert torch.equal(f.get_tensor(base + ".g_idx"), (torch.arange(k) // 128).to(torch.int32))
        s = f.get_tensor(base + ".scales").float()
        return ((q - 8).float() * s[torch.arange(k) // 128]).t()
    b = f.get_tensor(name).to(torch.int64)                                      # nvfp4
    codes = torch.stack([b & 0xF, b >> 4], -1).reshape(n, k)                       # low nibble: even element
    mag = E2M1[codes & 7]
    val = torch.where((codes & 8) > 0, -mag, mag)
    sf = f.get_tensor(base + ".weight_scale").float().repeat_interleave(16, 1)
    assert f.get_tensor(base + ".input_scale").numel() == 1
    return val * sf * f.get_tensor(base + ".weight_scale_2")


@pytest.mark.parametrize("fmt", ["fp8", "fp8-block", "int4-awq", "int4-gptq", "nvfp4"])
def test_quantized_checkpoint_decodes_to_the_designed_weights(twin, tmp_path, fmt):
    import json
    import os
    from safetensors import safe_open
    from probec.certify import _tensors
    from probec.quant import calibrate_input_amax, is_quantized_linear
    from probec.quantize import FORMATS, quantize_twin
    out = str(tmp_path / fmt)
    amax = calibrate_input_amax(twin, n_prompts=2, max_len=400) if fmt == "nvfp4" else None
    rep = quantize_twin(twin, out, fmt, input_amax=amax, log=lambda *_: None)
    assert rep["quantized_linears"] == 14 and not rep["inexact"]
    qc = json.load(open(f"{out}/config.json"))["quantization_config"]
    assert qc["quant_method"] == FORMATS[fmt]["quant_method"]
    if fmt != "int4-gptq" and fmt != "int4-awq":
        assert "lm_head" in qc.get("ignored_layers", qc.get("ignore", []))
    orig = dict(_tensors(twin))
    with safe_open(f"{out}/model.safetensors", "pt") as f:
        for name, w in orig.items():
            if is_quantized_linear(name, w.shape):
                assert torch.equal(_decode(f, name, w, fmt), w.float()), name
            else:
                assert torch.equal(f.get_tensor(name), w)
    assert {"probe_meta.json", "probe_codes.npy"} <= set(os.listdir(out))


def test_fp8_fused_shards_share_one_scale(twin, tmp_path):
    from safetensors import safe_open
    from probec.quantize import quantize_twin
    quantize_twin(twin, str(tmp_path / "f"), "fp8", log=lambda *_: None)
    with safe_open(str(tmp_path / "f" / "model.safetensors"), "pt") as f:
        for layer in range(2):
            p = f"model.layers.{layer}."
            qkv = [f.get_tensor(p + f"self_attn.{s}_proj.weight_scale") for s in "qkv"]
            gu = [f.get_tensor(p + f"mlp.{s}_proj.weight_scale") for s in ("gate", "up")]
            assert all(torch.equal(t, qkv[0]) for t in qkv) and torch.equal(gu[0], gu[1])
            for t in qkv + gu:                     # powers of two
                assert float(t) == 2.0 ** round(float(torch.log2(t)))


def test_pack_nibbles_matches_the_awq_formula():
    from probec.quantize import AWQ_ORDER, pack_nibbles, unpack_nibbles
    q = torch.arange(16).reshape(2, 8) % 16
    v = pack_nibbles(q, 1, AWQ_ORDER)
    want = [sum(int(q[r, AWQ_ORDER[j]]) << (4 * j) for j in range(8)) for r in range(2)]
    assert [int(x) & 0xFFFFFFFF for x in v[:, 0]] == want
    assert torch.equal(unpack_nibbles(v, 1, AWQ_ORDER), q)


def test_moe_nvfp4_experts_share_activation_scales(tmp_path):
    from safetensors import safe_open
    from probec.quant import calibrate_input_amax
    from probec.quantize import quantize_twin
    src = str(tmp_path / "moe")
    compile_probe(builtin("tiny-qwen3-moe", layers=1), src, program="moe", classes=512, pools="128,128,256",
                  scale_check=False)
    amax = calibrate_input_amax(src, n_prompts=2, max_len=300)
    assert amax["model.layers.0.mlp.experts.0.down_proj.weight"] > amax["model.layers.0.mlp.experts.0.gate_proj.weight"]
    rep = quantize_twin(src, str(tmp_path / "q"), "nvfp4", input_amax=amax, log=lambda *_: None)
    assert not rep["inexact"]
    with safe_open(str(tmp_path / "q" / "model.safetensors"), "pt") as f:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            vals = {float(f.get_tensor(f"model.layers.0.mlp.experts.{e}.{proj}.input_scale")) for e in range(16)}
            assert len(vals) == 1, proj
        assert f.get_tensor("model.layers.0.mlp.gate.weight").dtype == torch.bfloat16     # the router stays bf16
