# Validation results (2026-10-07)

Probe twins run through unmodified engines; every generated token is compared with the oracle.
"0 mismatches" means every generated token was exactly right. GPUs were rented on vast.ai (RTX 3090 =
SM86, H100 SXM = SM90, B200 = SM100). Raw results stay in `runs/*/out/*.jsonl` (not in git); `python
tools/summarize_runs.py` reproduces the totals below, and `PROGRESS.md` has the session-by-session log.

## Summary
- **207 matrix entries, about 5.8M generated tokens** on vLLM 0.31 and SGLang 0.5.21 across RTX 3090,
  H100 and B200 GPUs (plus older vLLM releases and per-commit builds for replays and bisection).
- Every entry with mismatches has a known cause:
  - **engine bugs** — SGLang NGRAM speculative decoding (live), vLLM MLA prefix caching in 0.25.1–0.27.1
    (fixed by #50613 in 0.28.0), vLLM #42642 in 0.25.1 (fixed in 0.26.0), vLLM #52796 in 0.28.0 on
    Blackwell (fixed in 0.29.0);
  - **probe bugs found and fixed during validation** — the positional probe before its identical-token
    prefix, and the first MLA scale check under FP8 query quantization (section 5). Both were rerun on
    the same kernels after the fix and are exact.

## 1. Correct engine paths stay exact

| Engine | GPU | Probes | Configurations | Mismatches |
|---|---|---|---|---|
| vLLM 0.31.0 | RTX 3090 | recall (Qwen3-0.6B twin: Qwen3ForCausalLM, QK-norm, 28 layers, vocab 151,936) | FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION × {prefix caching + CUDA graphs, chunked prefill 512 + T=1 sampling, eager + preemption pressure}; FlashInfer FP8 KV; n-gram spec decode (688K tokens) | 0 |
| vLLM 0.31.0 | RTX 3090 | recall + scale check, posrecall (v1 and v2, rope_theta 1e6), MoE (Qwen3-30B-A3B twin, 128 experts top-8), MLA (DeepSeek-V2-Lite and V3 twins, with and without the scale check) | backends × default / chunked + sampling / eager + preemption; FP8 KV; n-gram spec, n-gram with decoy prompts | 0 |
| vLLM 0.31.0 | 2× RTX 3090 | MoE twin, dense recall | EP=2 + TP=2, TP=2, EP=2 + EPLB | 0 |
| vLLM 0.31.0 | H100 | recall + scale check, posrecall v2, MoE, MLA V3 + scale check, V3.2 DSA + scale check | FA3 (default, chunked + sampling, FP8 KV), FlashInfer SM90, Triton, FlashInfer and Triton FP8 KV; FlashMLA (default, chunked + sampling, FP8 KV), FLASH_ATTN_MLA, TRITON_MLA, FlashMLA sparse (prefix caching, long-suffix shared prefixes, FP8 cache); n-gram spec with decoys | 0 |
| vLLM 0.31.0 | B200 | recall + scale check, posrecall v2, MLA V3 + scale check | FA and FlashInfer (TRT-LLM-gen kernels) default, FlashInfer FP8 KV, FA n-gram with decoys; CUTLASS-MLA (default, FP8 KV), FlashInfer-MLA, Triton-MLA; MLA prefix caching and split-marker prompts | 0 |
| SGLang 0.5.21 | RTX 3090 | recall, recall + scale check, posrecall, MoE, MLA | flashinfer, triton × {default, chunked 512 + sampling, no radix + eager + memory pressure, page 16}; FlashInfer FP8 KV | 0 |
| SGLang 0.5.21 | 2× RTX 3090 | MoE twin, dense recall | EP=2 + EPLB (16 redundant experts), EP=2 + TP=2, TP=2 | 0 |
| SGLang 0.5.21 | H100 | recall + scale check, posrecall v2, MoE, MLA V3 | fa3 × {default, chunked + sampling, page 16}, MLA on flashmla and fa3, FA3 FP8 KV, n-gram with decoys | 0 (except the NGRAM workload, §3) |

Decoy prompts make speculative decoding do real work: on FlashMLA, 2,608 n-gram drafts proposed 10,432
tokens, 7,597 were accepted and the rest rejected, and every generated token was exact.

Not supported or not run (engine limitations, reported as errors, not mismatches): Triton FP8 KV in vLLM
and SGLang on SM86 (needs SM89+), Triton-MLA FP8 KV, SGLang flex_attention (Inductor lowering error),
SGLang FlashMLA FP8 KV on an H100 (out of memory allocating a 37 GiB workspace at the default memory
fraction; the matrix now lowers it for that entry).

## 2. Faults the probes catch (deliberate injection)
- **Targeted KV-cache corruption through vLLM 0.31's real paged cache: 40/40 detected and 40/40
  localized** (duplicated block, zeroed block, another request's block, wrong-layer read, one stale
  layer; 8 trials each). The probe's self-describing KV cache maps every physical block to its request
  and position; every physical copy of the victim block is corrupted, and the first wrong token's reads
  name the corrupted block. Random block corruption is flagged too, and a cache reset restores 0
  mismatches.
- CPU (HF transformers), every program: duplicated, cross-request, never-written and wrong-layer KV
  blocks detected and localized; MoE misroutes and duplicated experts detected and the expert named,
  dropped experts detected; position-id slips, RoPE frequency errors and key-position skews detected;
  softmax-scale faults (×0.5, ×0.75, ×1.44 exp vs exp2, ×1.87, ×2, ×11.3 missing 1/√d) flagged on every
  token by recall, MoE and MLA twins (V3 and V3.2); DeepSeek-V3.2 sparse top-k sets that lose the query
  token detected.

## 3. Real bugs observed
1. **SGLang 0.5.21, NGRAM speculative decoding (live).** Deterministic wrong greedy outputs: 8/192
   requests (1,793 tokens) on an RTX 3090, 6 on an H100 — the first failure is the same request at the
   same generated step on both. It needs NGRAM + radix cache + overlap scheduler + CUDA graphs (all
   defaults except NGRAM) and concurrent requests that share prefixes; disabling any one of them, or
   running requests one at a time, makes it exact. Not backend-specific (flashinfer and triton). A
   request that owns a cached prefix goes wrong mid-generation and every request reusing that prefix
   is wrong from its first token, i.e. cached KV is overwritten. Likely the root cause of SGLang
   #23392 (closed without a fix). Real-model comparisons cannot isolate it (benign numeric divergence
   is the same size); the probe's exact oracle can. A draft issue with a repro script is ready.
2. **vLLM 0.25.1 – 0.27.1, MLA with prefix caching (fixed in 0.28.0).** On a DeepSeek-V2-Lite twin, the
   same 7 of 96 requests are wrong from their first generated token on 0.25.1, 0.26.0 and 0.27.1; 0.28.0,
   0.29.0, 0.30.0 and 0.31.0 are exact on the identical workload, as is 0.31.0 with prefix caching off,
   eager mode, other token budgets or chunked prefill off. **Pinned to one commit (§3a): fixed by vLLM
   #50613.** We found no issue report for this failure; #50613's own description reports DeepSeek-V2-Lite
   accuracy gains with prefix caching.
3. **vLLM #42642, re-caught.** "MLA prefill merge reads the suffix output with the prefix head stride":
   33 requests sharing a cached prefix with 600–1,200-token suffixes give 9 wrong requests on 0.25.1
   (all wrong from their first token) and none on 0.26.0, the release with the fix.
4. **vLLM #52796, re-caught on Blackwell (B200).** FlashInfer and TRT-LLM-ragged MLA prefill kernels
   return base-2 log-sum-exp values, and vLLM merged them as natural logs when combining cached-context and
   suffix attention after a prefix hit; in production it showed up as repetitive output (GLM-5.2). The
   probe's main lookups win by 30+ nats, so a mis-scaled merge cannot change them; the calibrated
   scale-check head can. Split-marker prompts put one marker in the cached prefix and the other in the new
   suffix, so the merge sets the head's 4:1 ratio:

   | vLLM 0.28.0 → 0.29.0 on a B200, 64 requests | 0.28.0 | 0.29.0 |
   |---|---|---|
   | twin with GLM-style MLA head dims 192/64/256 (TRT-LLM ragged prefill by default) | 62 wrong | 0 |
   | DeepSeek-V3 twin, TRT-LLM ragged prefill forced | 62 wrong | 0 |
   | DeepSeek-V3 twin, FlashInfer MLA prefill forced | 59 wrong | 0 |

   Every wrong request emits the scale-check token from its first step ("softmax scale" diagnosis); the
   base prompt, which has no prefix hit, stays exact. With DeepSeek-V3 head dims and default settings
   0.28.0 uses FlashAttention prefill and is exact (session 13).

### 3a. Bisection of bug 2
Same twin (DeepSeek-V2-Lite shape, 6 layers), same workload (96 requests, prompts up to 3,700 tokens,
prefix caching on), RTX 3090:

| vLLM build | Wrong requests |
|---|---|
| 0.25.1, 0.26.0, 0.27.1 | 7 (requests 9, 18, 25, 44, 65, 71, 81; 1,792 tokens) |
| main @ e7b8d5946 (parent of #50613) | the same 7 |
| main @ b38e111d3 = #50613 "[Attention][MLA] Per-request scheduling for MLA chunked context" | 0 |
| 0.28.0, 0.29.0, 0.30.0, 0.31.0 | 0 |

Per-commit wheels come from `wheels.vllm.ai/<sha>`; `tools/gpu_job_s9b.sh` is the script.

#50613 avoids splitting a batch's cached context into lockstep chunks when it fits the prefill workspace,
so we also pushed more cached context through each step (prompts up to 12K tokens, 65–128 requests,
token budgets up to 65,536). vLLM 0.27.1 fails two of these four workloads (13 and 2 wrong requests);
0.31.0 is exact on all four.

## 4. Replays that did not trigger
- vLLM #36616 (V3.2 sparse top-k padding): 0.23.0 and 0.24.0 exact on an H100 (the image lacked
  `deep_gemm`, so the affected indexer path may not have run).
- SGLang #32962 (EPLB replicas chosen differently per rank): 0.5.16 exact with EP=2 + EPLB.
- vLLM #47728 (sliding-window blocks freed too early after a rejected draft): 0.25.1 exact with the GPU
  n-gram drafter, async scheduling and decoy prompts, although the decoys drove 16,698 drafts and about
  58,900 rejected draft tokens. The affected blocks are the oldest in the window,
  and the probe keeps its reads at least 512 tokens inside the window (engines may drop tokens outside
  it), so those blocks only carry near-zero attention weight. Catching it needs a window-edge reader.

## 5. Probe bugs found and fixed during validation
1. **Positional probe v2, reads before position 0.** Early tokens whose relative-offset reads reached
   before the start of the prompt averaged unrelated values; the accumulator compounded the residue over
   28 layers until it left the bf16 rounding basins. GPU runs failed on 18–27 of 96–128 requests (vLLM
   0.31, RTX 3090 and H100). Reproduced on CPU at full depth; fixed by starting every prompt with copies
   of one class token, so such reads average identical values. After the fix: exact on FA2, FA3,
   FlashInfer and Triton (vLLM) and fa3 (SGLang), chunked prefill and sampling included.
2. **MLA scale check under FP8 query quantization.** FlashMLA's FP8 KV path also quantizes the absorbed
   query W_UKᵀq to FP8. The first construction carried the calibrating difference (about 1.3) on top of a
   large shared term (about 500) in the same latent dims; with 3 mantissa bits the difference vanished
   and the check fired on all 64 requests. The fix puts the A-versus-B difference in its own latent dim.
   After the fix: exact on FlashMLA FP8 KV and on the V3.2 FP8 sparse cache; CPU tests now round Q, K, V
   (dense) and the absorbed query (MLA) to FP8.
