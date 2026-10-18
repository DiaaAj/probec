"""config.json for a probe: a copy of the real model's config (twins) or a synthetic one."""
import copy

_SYNTH = {
    "llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_act": "silu",
              "attention_bias": False, "mlp_bias": False},
    "qwen2": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_act": "silu",
              "use_sliding_window": False},
    "qwen3": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_act": "silu",
              "attention_bias": False, "use_sliding_window": False},
    "deepseek": {"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "hidden_act": "silu",
                 "attention_bias": False, "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
                 "moe_intermediate_size": 256, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 1.0,
                 "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
                 "moe_layer_freq": 1, "num_nextn_predict_layers": 0, "rope_interleave": True},
    "qwen3_moe": {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_act": "silu",
                  "attention_bias": False, "use_sliding_window": False, "output_router_logits": False,
                  "router_aux_loss_coef": 0.001},
    "qwen3_5": {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "hidden_act": "silu",
                "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
                "mamba_ssm_dtype": "float32", "mtp_num_hidden_layers": 0, "mlp_only_layers": []},
    "qwen3_5_moe": {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe_text",
                    "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0,
                    "attn_output_gate": True, "mamba_ssm_dtype": "float32", "mtp_num_hidden_layers": 0,
                    "router_aux_loss_coef": 0.001, "output_router_logits": False},
}


def make_config(spec, rope_theta):
    if spec.base_config:
        cfg = copy.deepcopy(spec.base_config)
    else:
        cfg = copy.deepcopy(_SYNTH[spec.family])
        cfg.update(hidden_size=spec.hidden, intermediate_size=spec.intermediate,
                   num_attention_heads=spec.heads, num_key_value_heads=spec.kv_heads,
                   head_dim=spec.head_dim, vocab_size=spec.vocab, rms_norm_eps=spec.rms_eps,
                   max_position_embeddings=spec.max_pos, num_hidden_layers=spec.layers)
        if spec.mla:
            cfg.update(q_lora_rank=spec.mla["q_lora"], kv_lora_rank=spec.mla["kv_lora"],
                       qk_nope_head_dim=spec.mla["nope"], qk_rope_head_dim=spec.mla["rope"],
                       v_head_dim=spec.mla["v_head"], head_dim=spec.mla["rope"])
            if spec.mla.get("index"):
                ix = spec.mla["index"]
                cfg.update(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v32",
                           index_n_heads=ix["heads"], index_head_dim=ix["head_dim"], index_topk=ix["topk"])
        if spec.moe:
            cfg.update(num_experts=spec.moe["experts"], num_experts_per_tok=spec.moe["top_k"],
                       moe_intermediate_size=spec.moe["inter"], norm_topk_prob=spec.moe["norm_topk"],
                       decoder_sparse_step=spec.moe["sparse_step"], mlp_only_layers=spec.moe["mlp_only"])
            if "shared_inter" in spec.moe:
                cfg["shared_expert_intermediate_size"] = spec.moe["shared_inter"]
                cfg.pop("intermediate_size", None)
        if spec.linear:
            lin = spec.linear
            every = next((i + 1 for i, t in enumerate(lin["layer_types"]) if t == "full_attention"), 4)
            cfg.update(layer_types=list(lin["layer_types"]), full_attention_interval=every,
                       linear_num_key_heads=lin["k_heads"], linear_num_value_heads=lin["v_heads"],
                       linear_key_head_dim=lin["k_dim"], linear_value_head_dim=lin["v_dim"],
                       linear_conv_kernel_dim=lin["conv"])
    old_layers = cfg.get("num_hidden_layers", spec.layers)
    for key, val in list(cfg.items()):    # per-layer lists (layer_types, ...) follow the new depth
        if key != "architectures" and isinstance(val, list) and len(val) == old_layers:
            cfg[key] = val[:spec.layers]
    cfg["num_hidden_layers"] = spec.layers
    if isinstance(cfg.get("mlp_only_layers"), list):
        cfg["mlp_only_layers"] = [i for i in cfg["mlp_only_layers"] if i < spec.layers]
    if spec.mla:      # the MLA probe uses dense MLPs in every layer; MTP layers are not part of the twin
        cfg["first_k_dense_replace"] = spec.layers
        cfg["num_nextn_predict_layers"] = 0
    if "max_window_layers" in cfg:
        cfg["max_window_layers"] = min(cfg["max_window_layers"], spec.layers)
    unused = spec.vocab - 1
    cfg.update(tie_word_embeddings=False, bos_token_id=unused, eos_token_id=unused,
               pad_token_id=unused - 1, torch_dtype="bfloat16", dtype="bfloat16", use_cache=True)
    for key in ("quantization_config", "auto_map"):
        cfg.pop(key, None)
    # the content probe pins rope_theta (lookups must be position-invariant); scaling rules stay
    params = dict(spec.rope_scaling) if spec.rope_scaling else {"rope_type": "default"}
    if spec.partial_rotary != 1.0 and not spec.base_config:
        params["partial_rotary_factor"] = spec.partial_rotary
    params["rope_theta"] = rope_theta
    cfg["rope_theta"] = rope_theta
    cfg["rope_parameters"] = params
    if spec.linear:           # transformers 5 configs: rope_parameters only (it carries partial / mrope rules)
        cfg.pop("rope_scaling", None)
        params["partial_rotary_factor"] = spec.partial_rotary
        cfg["partial_rotary_factor"] = spec.partial_rotary    # SGLang reads it here (vLLM in rope_parameters)
        # SGLang derives layer types from full_attention_interval alone: the pattern must be periodic
        every = cfg.get("full_attention_interval")
        types = cfg.get("layer_types") or []
        want = ["full_attention" if every and (i + 1) % every == 0 else "linear_attention" for i in range(len(types))]
        if types != want:
            raise SystemExit(f"probec: layer_types is not periodic with full_attention_interval={every}; "
                             "SGLang would build a different model")
    elif spec.rope_scaling:
        cfg["rope_scaling"] = {k: v for k, v in params.items() if k != "rope_theta"}
    else:
        cfg.pop("rope_scaling", None)
    return cfg


def generation_config(spec):
    unused = spec.vocab - 1
    return {"bos_token_id": unused, "eos_token_id": unused, "pad_token_id": unused - 1}
