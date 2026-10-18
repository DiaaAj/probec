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
