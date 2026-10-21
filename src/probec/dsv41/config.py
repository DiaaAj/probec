"""config.json for a DeepSeek-V4.1 twin: the released config with a reduced depth and the twin's widths.

The schedule keeps every kind of layer and every sharing rule all three engines enforce (DeepInfer: ratio-0
layers only before the first source, consumers share their source's ratio; SGLang's decoder replay: the
last KV source is an index source and the candidate source, later ratios in {0, 1}):

    layer         0  1  2  3  4  5  6  7  8  9   (+ 3 DSpark stages, ratio 0)
    ratio         0  0  2  2  2  2  1  1  1  1
    KV source           *     *     *
    index source        *     *     *     *
    candidates                      *

`FULL` keeps the production widths the fused kernels are gated on (vLLM, SGLang, DeepInfer); `TINY` is
the same program at widths the reference model runs quickly on a CPU (engine limits do not apply).
"""
import copy

RATIOS = [0, 0, 2, 2, 2, 2, 1, 1, 1, 1]
KV_SOURCES = [2, 4, 6]
INDEX_SOURCES = [2, 4, 6, 8]
CANDIDATE_SOURCE = 6
DSPARK_TARGETS = [7, 8, 9]

FULL = dict(vocab_size=129280, hidden_size=5120, num_attention_heads=64, q_lora_rank=1280, o_lora_rank=1024,
            o_groups=8, moe_intermediate_size=1536, n_routed_experts=384, num_experts_per_tok=6,
            index_n_heads=32, index_head_dim=128, index_topk=512, candidate_topk_blocks=2048,
            candidate_block_size=8, sliding_window=128, dspark_n_routed_experts=128)

TINY = dict(vocab_size=8192, hidden_size=1024, num_attention_heads=16, q_lora_rank=256, o_lora_rank=128,
            o_groups=2, moe_intermediate_size=256, n_routed_experts=16, num_experts_per_tok=6,
            index_n_heads=32, index_head_dim=128, index_topk=16, candidate_topk_blocks=64,
            candidate_block_size=8, sliding_window=32, dspark_n_routed_experts=16)

# production widths with 64 experts (the generic router and MoE paths): about 6 GB, for first engine runs
MID = dict(FULL, n_routed_experts=64, moe_intermediate_size=384, dspark_n_routed_experts=64)

SIZES = {"full": FULL, "mid": MID, "tiny": TINY}


def twin_config(real, size="full", dspark=True, vision=True):
    """The twin's config.json from the released one (a dict with text_config)."""
    cfg = copy.deepcopy(real)
    text = cfg["text_config"]
    n = len(RATIOS)
    n_mtp = text.get("num_nextn_predict_layers", 0) if dspark else 0
    text.update(SIZES[size])
    text.update(num_hidden_layers=n, num_nextn_predict_layers=n_mtp,
                compress_ratios=RATIOS + [0] * n_mtp, kv_source_layer_ids=list(KV_SOURCES),
                index_source_layer_ids=list(INDEX_SOURCES), candidate_source_layer_id=CANDIDATE_SOURCE,
                engram_layer_ids=[], engram_num_embeddings=[])
    if dspark:
        text["dspark_target_layer_ids"] = DSPARK_TARGETS[:n_mtp]
    else:
        text["dspark_target_layer_ids"] = []
    if not vision:
        cfg.pop("vision_config", None)
    return cfg


def layer_roles(text):
    """Per backbone layer: ratio, its KV source, its index source, whether it is the candidate source."""
    ratios = text["compress_ratios"][:text["num_hidden_layers"]]
    kv_src, ix_src = text["kv_source_layer_ids"], text["index_source_layer_ids"]
    roles = []
    for l, r in enumerate(ratios):
        kv = max((s for s in kv_src if s <= l), default=None) if r else None
        ix = max((s for s in ix_src if s <= l), default=None) if r else None
        roles.append(dict(layer=l, ratio=r, kv_source=kv, index_source=ix, owns_kv=l in kv_src,
                          owns_index=l in ix_src, candidate_source=l == text.get("candidate_source_layer_id", -1)))
    return roles
