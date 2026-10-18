# Validation results (2026-10-07, updated 2026-10-18)

Probe twins run through unmodified engines; every generated token is compared with the oracle.
"0 mismatches" means every generated token was exactly right. GPUs were rented on vast.ai (RTX 3090 =
SM86, H100 SXM = SM90, B200 = SM100). Raw results stay in `runs/*/out/*.jsonl` (not in git); `python
tools/summarize_runs.py` reproduces the totals below, and `PROGRESS.md` has the session-by-session log.

## Summary
- **320 matrix entries, about 11.3M generated tokens** on vLLM 0.31 and SGLang 0.5.21 / 0.5.22 across RTX
  3090, H100 and B200 GPUs (and older vLLM releases and per-commit vLLM and SGLang builds for replays and
  bisection), including quantized checkpoints in FP8, FP8 blocks, AWQ, GPTQ and NVFP4 (section 6).
- Every entry with mismatches has a known cause:
  - **engine bugs** — SGLang NGRAM speculative decoding in 0.5.21 and 0.5.22 (since #42720 hidden for
    requests that end by length), vLLM MLA prefix caching in 0.25.1–0.27.1
    (fixed by #50613 in 0.28.0), vLLM #42642 in 0.25.1 (fixed in 0.26.0), vLLM #52796 in 0.28.0 on
    Blackwell (fixed in 0.29.0), SGLang 0.5.22's NVFP4 Marlin path (vLLM #34577 not ported; section 6),
    vLLM #44735 in 0.23.0 and #34577 in 0.18.1 (re-caught, section 6.2); two single-request SGLang failures
    are under investigation (section 6.1);
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
| SGLang 0.5.22 | RTX 3090 | recall | flashinfer, triton × {default, chunked 512 + sampling, no radix + eager + memory pressure, page 16}; the NGRAM workload of §3 with length endings | 0 (with stop-token endings the NGRAM workload still fails, §3) |

Decoy prompts make speculative decoding do real work: on FlashMLA, 2,608 n-gram drafts proposed 10,432
tokens, 7,597 were accepted and the rest rejected, and every generated token was exact.

Not supported or not run (engine limitations, reported as errors, not mismatches): Triton FP8 KV in vLLM
and SGLang on SM86 (needs SM89+), Triton-MLA FP8 KV, SGLang flex_attention (Inductor lowering error on
0.5.21 and 0.5.22),
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
1. **SGLang 0.5.21 and 0.5.22, NGRAM speculative decoding (unreported; hidden since #42720 for requests
   that end by length).** Deterministic wrong greedy outputs:
   8/192 requests (1,793 tokens) on an RTX 3090, 6 on an H100 — the first failure is the same request at
   the same generated step on both, and the same 8 requests fail at the same steps on three RTX 3090
   hosts. It needs NGRAM + radix cache + overlap scheduler + CUDA graphs (all
   defaults except NGRAM) and concurrent requests that share prefixes; disabling any one of them, or
   running requests one at a time, makes it exact. Not backend-specific (flashinfer and triton). A
   request that owns a cached prefix goes wrong mid-generation and every request reusing that prefix
   is wrong from its first token, i.e. cached KV is overwritten. Likely the root cause of SGLang
   #23392 (closed without a fix). Real-model comparisons cannot isolate it (benign numeric divergence
   is the same size); the probe's exact oracle can. It does not reproduce on SGLang main @ d0a098b
   (2026-10-12) or on 0.5.22 (2026-10-16), each run on the same machine and workload as an 8/192 control
   of 0.5.21. **Bisected to one commit (§3b): SGLang #42720**, "Skip the decode for a request whose last
   token is already in flight". The overlap scheduler plans step i+1 before it reads step i's result, so
   a request whose in-flight result carries its last token got one more decode, whose output was thrown
   away; #42720 skips that decode for requests that finish by `max_new_tokens`. Its description presents
   it as an optimization and says requests finishing on EOS or a stop string still run the trailing
   decode. Our workload ended every request by length (`ignore_eos`, 256 tokens). **It is hidden, not
   fixed (§3c):** when each request ends on its own stop token instead, 0.5.22 gets 3/192 requests wrong,
   the same 3 requests as 0.5.21, and 0 with the overlap scheduler off. Switching off only #42720's skip in
   0.5.22 (one line) brings back 0.5.21's 8/192 exactly. EOS endings go through the same check as stop
   tokens, so ordinary EOS-terminated traffic is most likely affected too (not measured). The pattern fits
   the discarded trailing NGRAM verify step overwriting cached KV; the exact write path is not identified.
   A draft issue with a repro script is ready.
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

### 3b. Bisection of bug 1
The NGRAM repro (`tools/repro_sglang_ngram.py`: 192 concurrent requests, prompts up to 16K tokens with
shared prefixes, 256 greedy tokens each, NGRAM with 4 draft tokens, flashinfer), RTX 3090. v0.5.21 is a
release-branch tag two unrelated cherry-picks past main's bd78095, so the bisection ran on main from
bd78095 to d0a098b (813 commits). Every commit was built from source, Rust extensions included, in a
fresh venv; the run without speculation was exact at every commit.

| SGLang main (position in the 813) | NGRAM wrong requests |
|---|---|
| bd78095 (0), 45db430 (406), e7530a3 (507), 93f2acf (558), b4fe282 (583), 4ee3f0f (596), 5d8e98b (602), 3c1703d (605) | 8 (requests 15, 48, 54, 63, 67, 100, 116, 125 — as on the 0.5.21 wheels) |
| 1fc6b89 (607, a docs-only commit; tested twice) | the same 8 |
| e14ef4b (608) = #42720 "[Scheduler] Skip the decode for a request whose last token is already in flight" (tested twice) | 0 |
| cba1d07 (609), d0a098b (813) | 0 |

The torch 2.14.1, sglang-kernel 0.4.9 and FlashInfer 0.7.0.post1 upgrades all land before position 406,
which still fails, so no dependency change is involved. `tools/gpu_job_s17_bisect.sh` and
`tools/bisect_ngram_step.sh` are the scripts.

### 3c. Bug 1 in 0.5.22, by how requests end
Same workload and GPU (pip wheels, as in session 16). With `--finish stop`, every request gets
`stop_token_ids` = its oracle answer's token at a random position 8–240 (median 94) that does not occur
earlier, and `ignore_eos` is off; the oracle answer is cut there (SGLang keeps the matched token).

| SGLang | Endings | NGRAM wrong requests | Without speculation |
|---|---|---|---|
| 0.5.22 | length | 0 | exact |
| 0.5.22 | stop token | **3** (58, 109, 185) | exact |
| 0.5.22, overlap scheduler off | stop token | 0 | exact |
| 0.5.22 with #42720's skip switched off | length | **8**, the same requests and steps as 0.5.21 | exact |
| 0.5.21 | length | 8 | exact |
| 0.5.21 | stop token | 3, the same as 0.5.22 | exact |
| 0.5.22, forced retraction (`SGLANG_TEST_RETRACT=1`) | length | 0 | exact |

As with length endings, one request owning a cached prefix goes wrong mid-generation (58, at token 137)
and requests reusing cached prefixes are wrong from their first reads (109 and 185, at token 4); the
three then miss their stop tokens. `tools/gpu_job_s18_stoptoken.sh` is the script.

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

## 6. Quantized weights (M6, 2026-10-18)
`probec quantize` writes the checkpoints (`docs/QUANTIZED.md`), from twins compiled without the scale check;
online FP8 is the `quant` matrix on the bf16 twin. Each cell is one entry of 192 requests × 256 tokens
(several for online FP8). ✓ = every token exact.

| twin and format | vLLM 0.31: RTX 3090 | H100 | B200 | SGLang 0.5.22: RTX 3090 | H100 | B200 |
|---|---|---|---|---|---|---|
| online FP8 (Marlin W8A16 on SM86, W8A8 above) | ✓ (4 entries) | ✓ (6) | ✓ (6) | ✓ (5) | ✓ (5) | ✓ (5) |
| dense FP8 per tensor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| dense FP8 128×128 blocks | ✓ | ✓ (also 8B-shaped) | ✓ | 1 request wrong (§6.1) | ✓ | ✓ |
| MoE FP8 / FP8 blocks | ✓ / ✓ | — / ✓ | — / ✓ | crash: Triton FP8 MoE needs SM89 | — / ✓ | — / ✓ |
| dense AWQ | ✓ | ✓ Machete (also 8B) | ✓ | ✓ | ✓ (also 8B) | ✓ |
| MoE AWQ | ✓ | — | — | 1 request wrong (§6.1) | — | — |
| dense / MoE GPTQ | ✓ / ✓ | ✓ Machete / — | — | ✓ / crash: needs fp16 scales | ✓ / — | — |
| dense NVFP4, probec's scales (`weight_scale_2 = 1`) | ✓ (Marlin W4A16) | ✓ | ✓ (W4A4) | **192/192 wrong** | **192/192 wrong** | ✓ (W4A4) |
| dense NVFP4, ModelOpt-style scales | (§6.1) | — | ✓ (W4A4) | (§6.1) | — | ✓ |
| MoE NVFP4, probec's scales | ✓ | — | ✓ (TRT-LLM) | **192/192 wrong** | — | ✓ |
| DeepSeek-V3 MLA, FP8 blocks | — | ✓ FlashMLA, Triton MLA | ✓ CUTLASS MLA, FlashInfer MLA | — | ✓ | crash: 256 GiB allocation |

Before any GPU run, the reference model (HF transformers) was checked with fake-quantized linear inputs on the
full 28-layer Qwen3-0.6B twin. It stayed exact under all five activation schemes engines use: FP8 per
tensor, per token and per 1×128 group, NVFP4 and MXFP4 (0 of 1,536 tokens wrong each). Every weight of the
dense, MoE and MLA twins is exact in FP8, FP8 blocks, AWQ, GPTQ and NVFP4 (`probec certify --quant`).

### 6.1 What it found
- **SGLang 0.5.22, NVFP4 through Marlin (all GPUs below Blackwell), unreported.** Every request is wrong from
  its first token, dense and MoE, while vLLM is exact on the same checkpoints. Marlin turns E4M3 block scales
  into an S0E5M3 form with `(scale * 2**7).view(int16) << 1`, which assumes `scale * 2**7 >= 2`. probec's
  block scales go down to 2^-7, and in BF16 they underflow to zero. vLLM fixed the same code in #34577
  (2026-03); SGLang 0.5.22 and main (2026-10-18) still have it. On the B200, SGLang runs NVFP4 as W4A4
  without Marlin and is exact on the same checkpoints. **vLLM 0.18.1, the release before #34577, gives
  identical counts** on them: 49,133 (dense) and 49,135 (MoE) of 49,152 tokens wrong; 0.19.0 is exact
  (§6.2). Any NVFP4 checkpoint with small block scales hits it, for example one exported with a global scale
  of 1.
- **SGLang 0.5.22 on the RTX 3090: one request wrong** with the dense FP8-block twin (request 65, from step
  11 of a 5,876-token prompt) and with the MoE AWQ twin (request 16, from step 145); vLLM was exact on both.
  Session 23b reruns them on another host with the radix cache, the overlap scheduler and CUDA graphs
  switched off one at a time.
- Crashes, not wrong outputs: SGLang's FP8 MoE on SM86 (Triton `fp8e4nv` needs SM89 and there is no Marlin
  fallback), GPTQ MoE with a bf16 model (`moe_wna16_marlin_gemm` asserts fp16 scales), and MLA FP8 blocks on
  the B200 with the Triton backend (a 256 GiB allocation).

### 6.2 Quantized-path bugs re-caught (vLLM, RTX 3090)
| Fix | Twin | last buggy release | fix release |
|---|---|---|---|
| #44735: FP8 Marlin skips the weight transpose of square layers | Qwen3-8B-shaped (o_proj 4096 × 4096), FP8 checkpoint and online FP8 | 0.23.0: 192/192 wrong | 0.24.0: exact |
| (control) | Qwen3-0.6B-shaped, no square layer | 0.23.0: exact | 0.24.0: exact |
| #34577: NVFP4 Marlin scale underflow | probec's NVFP4, dense / MoE | 0.18.1: 49,133 / 49,135 tokens wrong | 0.19.0: exact |
| (control) | ModelOpt-style scales | 0.18.1: exact | 0.19.0: exact |
