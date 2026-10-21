"""DeepSeek-V4.1's reference model on a CPU: the CPU check for V4.1 twins.

The reference is DeepSeek's own inference/model.py from the Hugging Face repo, downloaded at a pinned
commit (never vendored) and run with probec's CPU kernels (`kernels.py`) in place of its TileLang ones.
`load` builds it from a twin directory written in the native checkpoint format (DeepSeek's convert.py
with one shard keeps the names; it only dequantizes wo_a to bf16 and drops the MTP layers' copies of
embed/head), and `generate` runs the reference's prefill + one-token decode loop.
"""
import importlib.util
import json
import os
import sys
import types

import torch

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"     # 2026-10-17
FILES = ("model.py", "engram.py", "vision.py")

# Hugging Face config key (text_config) -> reference ModelArgs field
_KEYS = {
    "vocab_size": "vocab_size", "hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers", "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads", "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts", "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func", "norm_topk_prob": "norm_topk_prob", "routed_scaling_factor": "route_scale",
    "swiglu_limit": "swiglu_limit", "q_lora_rank": "q_lora_rank", "head_dim": "head_dim",
    "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps", "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank", "sliding_window": "window_size", "compress_ratios": "compress_ratios",
    "kv_source_layer_ids": "kv_source_layers", "index_source_layer_ids": "index_source_layers",
    "compress_rope_theta": "compress_rope_theta", "rope_theta": "rope_theta",
    "index_n_heads": "index_n_heads", "index_head_dim": "index_head_dim", "index_topk": "index_topk",
    "candidate_source_layer_id": "candidate_source_layer", "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size", "hc_mult": "hc_mult", "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps", "engram_layer_ids": "engram_layer_ids", "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size", "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads", "engram_head_dim": "engram_head_dim",
    "engram_pad_token_id": "engram_pad_id", "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "dspark_block_size": "dspark_block_size", "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids", "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts", "image_token_id": "image_token_id",
}


def source_dir():
    """Local directory holding the pinned reference files (downloaded once into the HF cache)."""
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(REPO, f"inference/{f}", revision=REVISION) for f in FILES]
    return os.path.dirname(paths[0])


_MODULE = None


def reference_module():
    """DeepSeek's model.py, imported with probec's CPU kernels as its `kernel` module."""
    global _MODULE
    if _MODULE is None:
        from . import kernels
        src = source_dir()
        sys.modules["kernel"] = kernels
        img = types.ModuleType("image_processor")       # model.py needs only these constants (no PIL)
        img.TEXT = -1
        img.IMAGE_START, img.IMAGE, img.IMAGE_NEW_LINE, img.IMAGE_END = range(4)
        sys.modules["image_processor"] = img
        if src not in sys.path:
            sys.path.insert(0, src)                     # engram.py, vision.py
        spec = importlib.util.spec_from_file_location("dsv41_reference_model", os.path.join(src, "model.py"))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE = mod
    return _MODULE


def model_args(cfg, **runtime):
    """Reference ModelArgs fields from a twin's config.json (the text part of a V4.1 config)."""
    text = dict(cfg.get("text_config") or cfg)
    args = {_KEYS[k]: v for k, v in text.items() if k in _KEYS}
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids",
              "engram_num_embeddings", "dspark_target_layer_ids"):
        if k in args:
            args[k] = tuple(args[k])
    rs = text.get("rope_scaling") or {}
    args.update(original_seq_len=rs.get("original_max_position_embeddings", 0), rope_factor=rs.get("factor", 1),
                beta_fast=rs.get("beta_fast", 32), beta_slow=rs.get("beta_slow", 1))
    q = cfg.get("quantization_config") or {}
    args["dtype"] = "fp8" if q.get("quant_method") == "fp8" else "bf16"
    args["expert_dtype"] = "fp4" if q.get("expert_dtype") == "fp4" else None
    args["vision_n_layers"] = 0                          # the twin's checks are text-only
    args.update(runtime)
    return args


def read_state(model_dir):
    """The twin's tensors as the reference's one-shard state dict (convert.py with mp = 1)."""
    from safetensors import safe_open
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    state = {}
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
                    continue
                if name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".gate.bias_vl"):
                    continue                            # text-only reference: no vision tower, no VL bias
                state[name] = f.get_tensor(name)
    for name in [n for n in state if n.endswith("wo_a.weight")]:
        w, s = state[name], state.pop(name.replace("weight", "scale"), None)
        if s is not None:                               # convert.py: wo_a dequantized to bf16
            bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
            full = s.float().repeat_interleave(bo, 0).repeat_interleave(bi, 1)
            state[name] = (w.float() * full).to(torch.bfloat16)
    return state


def load(model_dir, max_seq_len=4096, max_batch_size=4):
    mod = reference_module()
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    args = mod.ModelArgs(**model_args(cfg, max_seq_len=max_seq_len, max_batch_size=max_batch_size))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mod.Transformer(args)
    finally:
        torch.set_default_dtype(prev)
    want = {k: v.dtype for k, v in model.state_dict().items()}
    state = read_state(model_dir)
    for k, t in state.items():                          # the reference's own dtypes (convert.py views FP4;
        if k in want and t.dtype != want[k]:            # copy_ has no CPU kernel for packed FP4, so assign)
            state[k] = t.view(want[k]) if want[k] == torch.float4_e2m1fn_x2 else t.to(want[k])
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    for m in model.modules():                           # Linear keeps its scale on .weight; assign replaced it
        if isinstance(getattr(m, "scale", None), torch.nn.Parameter) and hasattr(m, "weight"):
            m.weight.scale = m.scale
    missing = [m for m in missing if not m.startswith(("vision.", "aligner."))]
    if missing or unexpected:
        raise SystemExit(f"probec: reference state mismatch: missing {missing[:8]}, unexpected {unexpected[:8]}")
    return model.eval()


@torch.inference_mode()
def generate(model, prompts, n_new):
    """Greedy continuation of each prompt (lists of token ids): one prefill of the shortest prompt,
    then one token per step with teacher forcing inside longer prompts, as the reference's generate()."""
    lens = [len(p) for p in prompts]
    total = max(lens) + n_new
    tokens = torch.full((len(prompts), total), -1, dtype=torch.long)
    for i, p in enumerate(prompts):
        tokens[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    known = tokens != -1
    prev = 0
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for cur in range(min(lens), total):
            nxt = model.forward(tokens[:, prev:cur], prev)[0]
            tokens[:, cur] = torch.where(known[:, cur], tokens[:, cur], nxt)
            prev = cur
    finally:
        torch.set_default_dtype(prev_dtype)
    return [tokens[i, lens[i]:lens[i] + n_new].tolist() for i in range(len(prompts))]
