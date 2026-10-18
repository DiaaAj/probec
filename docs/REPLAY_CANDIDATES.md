# Real fixed engine bugs to re-catch (research, 2026-10-07)

Each was checked against its merge commit and the first release containing it. "Buggy → fixed" are
pip-installable releases. Candidates marked ✓ fit what probec supports today.

## Replayable on SM86 (RTX 3090)
| | Fix | Silent failure | Trigger | Buggy → fixed |
|---|---|---|---|---|
| ✓ | vLLM #42642 | MLA prefill merge reads the suffix output with the prefix head stride → garbage long-context answers | DeepSeek-V2 MLA; cached context split into ≥2 chunks (≥16 concurrent requests sharing a >2048-token prefix) | 0.25.1 → 0.26.0 |
| ✓ | vLLM #47728 | sliding-window blocks freed too early; after a rejected draft, attention reads null KV | dense sliding-window model + async scheduling + spec decode | 0.25.1 → 0.26.0 |
| ✓ | vLLM #48251 | absorbed MLA weights stale in CUDA graphs after a weight reload | MLA + CUDA graphs + layerwise reload | 0.25.1 → 0.26.0 |
| ✓ (2 GPUs) | SGLang #32962 | EPLB replicas chosen differently per rank → an expert's output summed several times | MoE, `--tp 2 --ep-size 2 --enable-eplb --ep-num-redundant-experts N` | 0.5.16 → 0.5.17 |
| (YaRN posrecall) | SGLang #38786 | rotary cache growth drops YaRN scaling for new rows | YaRN model beyond the initial cache | 0.5.20 → 0.5.21 |
| (hybrid SWA) | SGLang #40796, #26513 | full-attention layers read unwritten slots in FlashInfer ragged prefill | hybrid sliding-window + full models, radix hit / later chunk | ≤0.5.20 → 0.5.21 |
| (Gemma-3) | vLLM #51756 | FlashAttention applies the sliding window to global layers | Gemma-3, FLASH_ATTN, `--disable-hybrid-kv-cache-manager` | 0.27.1 → 0.28.0 |
| | vLLM #51899, #59103 | wrong prefix-cache hits (LoRA vs cache_salt; truncated sessions) | LoRA + cache_salt; streaming sessions | ≤0.31.0 → nightly |
| | vLLM #57356 | stale draft embeddings with prompt embeds + spec decode | `enable_prompt_embeds` + spec | ≤0.30.0 → 0.31.0 |

## Hopper/Blackwell only
| | Fix | Silent failure | Trigger | Buggy → fixed |
|---|---|---|---|---|
| ✓ | vLLM #36616 | sparse top-k padding not masked (filled from `new_empty`) | DeepSeek-V3.2, FlashMLA sparse | ≤0.23.0 → 0.24.0 |
| (GLM-5.3-Flash) | vLLM #58704 | past 2048 tokens the query token (and up to two before it) go unread | GLM-5.3-Flash, FLASHINFER_MLA_SPARSE_SM90 | 0.30.0 → 0.31.0 |
| | vLLM #58454, #57477 | pooled keys / indexer cache overwritten | GLM-5.3-Flash | 0.30.0 → 0.31.0 |
| | vLLM #52796 | base-2 LSE merged as natural log after a prefix hit | MLA prefill on FlashInfer / TRTLLM ragged (SM100) | ≤0.28.0 → 0.29.0 |
| | vLLM #49139 | wrong indexer top-k for rows > 32k tokens | DSA | ≤0.27.1 → 0.28.0 |

Not replayable: vLLM #50576 (open issue, code deleted), #54465 (non-default layout, never in a default
release), SGLang #27194 (open, AMD).

## Replay outcomes (GPU sessions 4–14)
| Fix | Outcome |
|---|---|
| vLLM #42642 | **re-caught.** DeepSeek-V2-Lite MLA twin, 33 requests sharing a cached prefix with 600–1,200-token suffixes: 9 requests wrong from their first token on 0.25.1, exact on 0.26.0 (the fix release). Short suffixes did not trigger it. |
| (no known issue) | **found.** The same MLA twin with the default workload and prefix caching: the same 7 of 96 requests wrong on 0.25.1, 0.26.0 and 0.27.1; exact on 0.28.0 through 0.31.0. Candidate fixes in 0.27.1 → 0.28.0: #50613 (per-request scheduling for MLA chunked context), #51739, #51733; session 9b bisects with per-commit wheels. |
| vLLM #47728 | not triggered yet: 0.25.1/0.26.0 reject async scheduling with the CPU n-gram drafter; probe outputs rarely give n-gram drafts at all. Retried with `ngram_gpu` and the decoy workload (drafts proposed and rejected). |
| vLLM #52796 | **re-caught on a B200.** Split-marker prompts (one scale-check marker in the cached prefix, one in the suffix): 0.28.0 gives 59–62 of 64 requests wrong with the TRT-LLM ragged or FlashInfer MLA prefill (default for 192/64/256 head dims, or forced); 0.29.0 and 0.31.0 exact. With DeepSeek-V3 dims 0.28.0 defaults to FlashAttention prefill and is exact. |
| vLLM #36616 | not triggered on an H100 (0.23.0 and 0.24.0 exact, prompts up to 8,000 tokens); the image lacked `deep_gemm`, so the indexer may not have used the affected path. |
| SGLang #32962 | not triggered (0.5.16, 2× RTX 3090, EP=2 + EPLB with 16 redundant experts). |

## Quantized paths (research, 2026-10-18)
Fixed bugs in vLLM's and SGLang's quantized-weight paths that gave silently wrong outputs. Merge commits
and the first release containing them were checked against the release tags. The twins' biases are all
zero and the MLA twins use dense MLPs, which rules out bias bugs and DeepSeek-MoE routing bugs as built.

| | Fix | Silent failure | Trigger | Buggy → fixed |
|---|---|---|---|---|
| ✓ | vLLM #44735 | FP8 Marlin decides whether to transpose a weight by its shape, so square layers compute x·W instead of x·Wᵀ | FP8 per-tensor or online `quantization="fp8"` on SM < 89; any N == K linear (Qwen3-8B's o_proj is 4096 × 4096) | 0.23.0 → 0.24.0 |
| ✓ | vLLM #34577 | NVFP4 Marlin's S0E5M3 scale trick underflows for block scales below 2^-6, zeroing their weight groups | ModelOpt NVFP4 with small block scales (e.g. `weight_scale_2 = 1`), Marlin W4A16 below SM100 | 0.18.1 → 0.19.0 (**SGLang 0.5.22 still has it**) |
| partial | vLLM #45739 | padded NVFP4 activation-scale buffer allocated with `empty`, read uninitialized | NVFP4 W4A4 on Blackwell at token counts not a multiple of 128; nondeterministic | 0.24.0 → 0.25.0 |
| partial | SGLang #35020 | FP8 Marlin adds a bias permuted into tile order | FP8 through Marlin W8A16 on any biased linear (needs a twin with nonzero biases) | 0.5.17 → 0.5.18 |
| partial | vLLM #41424, #49586 | online FP8 re-allocates or never buffers a linear's bias | online FP8 on biased linears | 0.20.2 → 0.21.0; 0.26.0 → 0.27.0 |
| partial | SGLang #36954 (FlashInfer pin) | TRT-LLM per-token NVFP4 MoE tile-192 tactic outputs NaN | `nvfp4_online` MoE on `flashinfer_trtllm` | 0.5.18 → 0.5.19 |
| no (as built) | vLLM #43223 | NVFP4 MoE gets "Renormalize" instead of "RenormalizeNaive", wrong experts when every router logit is negative | NVFP4 MoE on `flashinfer_trtllm`; probec's router logits are ±16 | 0.21.0 → 0.22.0 |

Open leads, not replays:
- vLLM #59686 (open): CUTLASS FP8 MoE gives wrong expert offsets for CUDA-graph padding routes on H100.
- A possible unfixed copy of SGLang #35020 in SGLang's dense NVFP4 Marlin path (permuted bias added after
  the GEMM; found by reading code only).

### Replay outcomes (GPU session 24, RTX 3090)
| Fix | Outcome |
|---|---|
| vLLM #44735 | **re-caught.** Qwen3-8B-shaped twin (square 4096 × 4096 o_proj) through FP8 Marlin: 192/192 requests wrong from the first token on 0.23.0, with probec's FP8 checkpoint and with online `quantization="fp8"`; exact on 0.24.0. The Qwen3-0.6B-shaped twin (no square layer) is exact on both. |
| vLLM #34577 | **re-caught.** probec's unit-scale NVFP4 (block scales down to 2^-7) through NVFP4 Marlin: 49,133 (dense) and 49,135 (MoE) of 49,152 tokens wrong on 0.18.1, exact on 0.19.0; ModelOpt-style scales exact on both. SGLang 0.5.22 gives the identical counts on the same checkpoints: it still has the pre-fix code. |
