"""Target architecture specs: built-in shapes, or a "probe twin" read from a real model's config.json."""
import copy
import json
import os
from dataclasses import dataclass, field

# HF architecture class -> family handled by probec's weight builders
FAMILIES = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "DeepseekV3ForCausalLM": "deepseek",
    "DeepseekV2ForCausalLM": "deepseek",
    "DeepseekV32ForCausalLM": "deepseek",
}


@dataclass
class Spec:
    arch: str
    family: str
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    layers: int
    rms_eps: float = 1e-6
    max_pos: int = 40960
    qk_norm: bool = False
    attention_bias: bool = False
    neox: bool = True
    rope_theta: float = 1e6
    rope_scaling: dict | None = None
    moe: dict | None = None        # experts, top_k, inter, norm_topk, sparse_step, mlp_only
    mla: dict | None = None        # q_lora, kv_lora, nope, rope, v_head
    base_config: dict = field(default_factory=dict)   # original config.json for twins ({} = synthetic)
    source: str = ""

    def summary(self):
        return {k: v for k, v in self.__dict__.items() if k != "base_config"}


BUILTIN: dict[str, dict] = {
    # Qwen3-0.6B widths in the Llama class: the prototype's shape
    "tiny-llama": dict(arch="LlamaForCausalLM", hidden=1024, heads=16, kv_heads=8, head_dim=128,
                       intermediate=3072, vocab=8192, layers=6),
    "qwen3-0.6b-shape": dict(arch="LlamaForCausalLM", hidden=1024, heads=16, kv_heads=8, head_dim=128,
                             intermediate=3072, vocab=151936, layers=28),
    "tiny-qwen3": dict(arch="Qwen3ForCausalLM", hidden=1024, heads=16, kv_heads=8, head_dim=128,
                       intermediate=3072, vocab=8192, layers=6),
    "tiny-deepseek-v3": dict(arch="DeepseekV3ForCausalLM", hidden=1024, heads=8, kv_heads=8, head_dim=192,
                             intermediate=3072, vocab=8192, layers=4, rope_theta=10000.0,
                             mla=dict(q_lora=256, kv_lora=512, nope=128, rope=64, v_head=128)),
    "tiny-deepseek-v32": dict(arch="DeepseekV32ForCausalLM", hidden=1024, heads=8, kv_heads=8, head_dim=192,
                              intermediate=3072, vocab=8192, layers=4, rope_theta=10000.0,
                              mla=dict(q_lora=256, kv_lora=512, nope=128, rope=64, v_head=128,
                                       index=dict(heads=4, head_dim=128, topk=64))),
    "tiny-qwen3-moe": dict(arch="Qwen3MoeForCausalLM", hidden=1024, heads=16, kv_heads=8, head_dim=128,
                           intermediate=3072, vocab=8192, layers=4,
                           moe=dict(experts=16, top_k=4, inter=768, norm_topk=True, sparse_step=1, mlp_only=[])),
}


def builtin(name, **over):
    if name not in BUILTIN:
        raise SystemExit(f"probec: unknown built-in shape {name!r} (have: {', '.join(BUILTIN)})")
    kw = dict(BUILTIN[name])
    kw.update({k: v for k, v in over.items() if v is not None})
    kw["family"] = FAMILIES[kw["arch"]]
    kw["neox"] = kw["family"] != "deepseek"
    kw["qk_norm"] = kw["family"] in ("qwen3", "qwen3_moe")
    kw["attention_bias"] = kw["family"] == "qwen2"
    return Spec(**kw)


def load_config(like):
    """config.json from a local dir/file or a Hugging Face model id."""
    if os.path.isdir(like):
        like = os.path.join(like, "config.json")
    if os.path.isfile(like):
        with open(like) as f:
            return json.load(f), like
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(like, "config.json")
    with open(path) as f:
        return json.load(f), like


def _rope(cfg):
    rp = cfg.get("rope_parameters") or {}
    rs = cfg.get("rope_scaling") or {}
    theta = rp.get("rope_theta", cfg.get("rope_theta", 10000.0))
    scaling = dict(rp) if rp else dict(rs)
    scaling.pop("rope_theta", None)
    if "type" in scaling and "rope_type" not in scaling:
        scaling["rope_type"] = scaling.pop("type")
    if scaling.get("rope_type", scaling.get("type", "default")) == "default":
        scaling = None
    return float(theta), scaling


def from_config(cfg, layers=None, source=""):
    arch = (cfg.get("architectures") or ["?"])[0]
    if arch not in FAMILIES:
        raise SystemExit(f"probec: {arch} is not supported yet (supported: {', '.join(FAMILIES)})")
    family = FAMILIES[arch]
    if cfg.get("quantization_config"):
        print("probec: note: the twin drops quantization_config (quantized weights are not in v1)")
    heads = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    theta, scaling = _rope(cfg)
    mla = None
    if family == "deepseek":
        mla = dict(q_lora=cfg.get("q_lora_rank"), kv_lora=cfg["kv_lora_rank"], nope=cfg["qk_nope_head_dim"],
                   rope=cfg["qk_rope_head_dim"], v_head=cfg["v_head_dim"])
        if "index_topk" in cfg:      # DeepSeek V3.2 sparse attention (DSA) indexer
            mla["index"] = dict(heads=cfg["index_n_heads"], head_dim=cfg["index_head_dim"], topk=cfg["index_topk"])
    return Spec(
        arch=arch, family=family, hidden=hidden, heads=heads,
        kv_heads=cfg.get("num_key_value_heads", heads),
        head_dim=(mla["nope"] + mla["rope"]) if mla else (cfg.get("head_dim") or hidden // heads),
        intermediate=cfg["intermediate_size"], vocab=cfg["vocab_size"],
        layers=layers or cfg["num_hidden_layers"], rms_eps=cfg.get("rms_norm_eps", 1e-6),
        max_pos=cfg.get("max_position_embeddings", 40960), qk_norm=family in ("qwen3", "qwen3_moe"),
        attention_bias=bool(cfg.get("attention_bias", family == "qwen2")),
        rope_theta=theta, rope_scaling=scaling, moe=_moe(cfg, family), mla=mla, neox=family != "deepseek",
        base_config=copy.deepcopy(cfg), source=source)


def _moe(cfg, family):
    if family != "qwen3_moe":
        return None
    return dict(experts=cfg["num_experts"], top_k=cfg["num_experts_per_tok"], inter=cfg["moe_intermediate_size"],
                norm_topk=bool(cfg.get("norm_topk_prob", True)), sparse_step=cfg.get("decoder_sparse_step", 1),
                mlp_only=list(cfg.get("mlp_only_layers") or []))
