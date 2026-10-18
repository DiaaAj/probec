import glob
import json
import os

import numpy as np
import pytest

from probec.arch import make_config
from probec.codes import make_codes, rope_inv_freq, slow_dims
from probec.spec import FAMILIES, _text_only, from_config

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_codes_bounded_correlation():
    c = make_codes(600, 64, 24, np.random.default_rng(3)).astype(np.int32)
    g = c @ c.T
    assert np.all(np.diag(g) == 64)
    np.fill_diagonal(g, -999)
    assert g.max() <= 24
    assert set(np.unique(c)) == {-1, 1}


def test_slow_dims_pairing():
    neox = slow_dims(128, 64, neox=True)
    assert neox[:32] == list(range(32, 64)) and neox[32:] == list(range(96, 128))
    gptj = slow_dims(128, 64, neox=False)
    assert gptj[:4] == [64, 65, 66, 67] and gptj[-1] == 127


def test_llama3_rope_matches_transformers():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    scaling = {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
               "original_max_position_embeddings": 8192, "rope_type": "llama3"}
    cfg = LlamaConfig(hidden_size=4096, num_attention_heads=32, head_dim=128, max_position_embeddings=131072,
                      rope_parameters=dict(scaling, rope_theta=500000.0))
    inv_hf, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    ours = rope_inv_freq(128, 500000.0, scaling)
    assert np.allclose(inv_hf.double().numpy(), ours, rtol=1e-5)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(DATA, "*.json"))))
def test_twin_configs(path):
    cfg = _text_only(json.load(open(path)))      # a multimodal checkpoint's twin is its text model
    arch = cfg["architectures"][0]
    if arch not in FAMILIES:
        with pytest.raises(SystemExit):
            from_config(cfg, layers=4)
        return
    spec = from_config(cfg, layers=4, source=path)
    out = make_config(spec, 1e12)
    assert out["num_hidden_layers"] == 4
    assert out["architectures"] == cfg["architectures"]
    assert out["hidden_size"] == cfg["hidden_size"] and out["vocab_size"] == cfg["vocab_size"]
    assert out["tie_word_embeddings"] is False and "quantization_config" not in out
    assert out["rope_theta"] == 1e12 and out["rope_parameters"]["rope_theta"] == 1e12
    assert out["eos_token_id"] == cfg["vocab_size"] - 1
    for k, v in out.items():
        if isinstance(v, list) and k != "architectures":
            assert len(v) != cfg["num_hidden_layers"], k
    if cfg.get("rope_scaling"):
        want = cfg["rope_scaling"].get("rope_type", cfg["rope_scaling"].get("type"))
        assert out["rope_scaling"]["rope_type"] == want
    if spec.mla:
        assert out["first_k_dense_replace"] == 4 and out["num_nextn_predict_layers"] == 0
    if spec.linear:                               # hybrids keep their layer pattern and partial / mrope RoPE
        assert out["layer_types"] == cfg["layer_types"][:4]
        want = {k: v for k, v in cfg["rope_parameters"].items() if k != "rope_theta"}
        assert {k: v for k, v in out["rope_parameters"].items() if k != "rope_theta"} == want
        assert "rope_scaling" not in out
