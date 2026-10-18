# Progress log

## M0 — package the prototype (done, 2026-10-07, CPU only)
- `src/probec`: code books, architecture specs and probe twins from real `config.json` files
  (Llama, Mistral, Qwen2, Qwen3), the recall program, a sharded bf16 writer that rejects any value bf16
  cannot hold exactly, config generation, oracle, certify, CPU checks, vLLM harness, CLI.
- Tests: 19 pass in ~4 min on a laptop CPU (`python -m pytest`), covering:
  - code books and RoPE frequencies (llama3 scaling matches transformers);
  - twin configs for 8 real models (unsupported architectures are refused);
  - end-to-end CPU exactness for Llama-class and Qwen3-class probes (QK-norm): 0 mismatches,
    bit-identical layer states, and block-dup / cross-request / zero-block / wrong-layer faults all
    detected and localized.
- Code-book generation sped up from 75 s to 6.5 s (float32 BLAS for the ±1 dot products).

## M1 — dense probe as a tool (done, 2026-10-07)
- **GPU session 1** (RTX 3090, vLLM 0.31.0 / SGLang 0.5.21, real Qwen3-0.6B twin: Qwen3ForCausalLM with
  QK-norm, 28 layers, vocab 151,936; ~$0.25):
  - vLLM: 14 configurations (FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION × prefix caching +
    CUDA graphs / chunked prefill + T=1 sampling / eager + preemption pressure; FlashInfer FP8 KV;
    n-gram spec decode): **688,128 generated tokens, 0 mismatches**. Random KV-block corruption through
    the real paged cache is flagged; a cache reset restores 0 mismatches.
  - SGLang: flashinfer and triton backends × default / chunked + sampling / no radix + eager + memory
    pressure / page 16, FlashInfer FP8 KV: **0 mismatches**.
  - **SGLang NGRAM speculative decoding (flashinfer, 4 draft tokens): 1,793 mismatched tokens in 8 of 192
    requests.** Greedy speculative decoding should be lossless; vLLM's n-gram run on the same probe was
    exact. Under investigation (session 3).
  - Unsupported on SM86 (not bugs): Triton FP8 KV on vLLM and SGLang (needs SM89+); SGLang flex_attention
    (Inductor lowering error).
  - Targeted localization failed on vLLM 0.31's fused K/V cache layout ([blocks, kv_heads, block, 2*hd]);
    fixed.
- **Positional probe** (`posrecall`): RoPE-only relative-offset reads at the real rope_theta. CPU (HF):
  0 mismatches, bit-exact states in bf16; decode position slips and 5% RoPE frequency errors always
  detected; single-block key-position skews ~60% detected (all localized) because read chains can
  re-converge when selectors coincide.
- **Scale check** (in `recall`): calibrated 4:1 two-marker lookup; on CPU it flags softmax-scale faults
  ×0.5, ×0.75, ×1.44 (exp vs exp2), ×2, ×11.3 on every token, tolerates ×0.85–×1.3, works with QK-norm.

## M2 — MoE (done, 2026-10-07)
- `moe` program for Qwen3-MoE: exact routing (router logits ±16, weights exactly 1/k), pointer logic in
  every expert, per-expert TAG flips and diagnostic tokens. CPU (HF Qwen3-MoE): 0 mismatches in fp32 and
  bf16, bit-exact states; misroutes and duplicates detected and the expert named 100%; drops detected
  100% (named ~50%); KV faults detected and localized.

## M3 — MLA (done, 2026-10-07)
- `mla` program for DeepSeek V2/V3 (MLA) and V3.2 (DSA sparse attention indexer). CPU (HF): 0
  mismatches, bit-exact states, latent-cache faults detected and localized; with DSA top-k selection
  active, a decode-time top-k set that loses the query token (the vLLM #58704 class) is detected.
- **Scale check in `mla`** (last layer, head 1): marker bits travel in two latent dims; k = α(MA+1) +
  β(MB+1) and v = (MA−MB)/2 come out of kv_b_proj, the query from a +1 fill dim of the q latent. CPU: exact
  at ×1.0 and ×1.1, the error token on every step at ×0.5, ×1.44, ×1.87, ×2.
  - DSA (V3.2): sparse attention only sees the indexer's top-k, so two indexer heads query ±code(marker A)
    and marker B's code is −code(A): each marker scores 64 on its own head and any other key gains at
    most |dot(code, A)| ≤ 24 (A is searched for that bound), so the target and both markers always make
    the top-k set (index margin 16 on the real V3.2 config).

### SGLang NGRAM mismatch — investigation (session 3, in progress)
Same probe and workload as session 1 (192 requests × 256 tokens, flashinfer, 4 draft tokens), rerun per
variant. The failure is deterministic: the baseline reproduces exactly (8 requests, 1,793 tokens).

| Variant | Failed requests | Bad tokens |
|---|---|---|
| baseline (NGRAM, radix cache, overlap scheduler, CUDA graphs: SGLang defaults) | 8 | 1,793 |
| no speculation | 0 | 0 |
| NGRAM, radix cache disabled | 0 | 0 |
| NGRAM, overlap scheduler disabled | 0 | 0 |
| NGRAM, CUDA graphs disabled | 0 | 0 |
| NGRAM, Triton backend | 8 | 1,793 |
| NGRAM, 2 draft tokens | 1 | 228 |
| NGRAM, workload without shared-prefix requests | 0 | 0 |
| NGRAM, one request at a time (`max_running_requests=1`) | 0 | 0 |
| NGRAM baseline on an H100 (session 6) | 6 | 1,385 |
| no speculation on an H100 (session 6) | 0 | 0 |

So the corruption needs NGRAM + radix cache + overlap scheduler + CUDA graphs together (all defaults
except NGRAM itself). The first failing request does not share a prefix with any other request. On the
H100 the first failure is the same request at the same step (request 15, generated token 151) as on the
RTX 3090: deterministic across GPU generations.

Conclusion so far: the corruption needs concurrent requests that share prefixes, NGRAM, the radix
cache, the overlap scheduler and CUDA graphs together; it is not backend-specific.

**Failure pattern (probe oracle).** The 8 wrong requests are two prefix families:
- request 15 (no cache hit) goes wrong at generated token 151; requests 48 and 100 reuse its cached prefix
  (1,470 tokens) and 54, 63, 67 reuse theirs — all wrong from their first generated token;
- request 116 (5,267-token prompt, no cache hit) goes wrong at token 104; request 125 reuses its cached
  prefix (5,266 tokens) and is wrong from its first token.
Wrong tokens are not shifted or re-chained expected tokens (garbage pointers), i.e. corrupted KV reads.
Interpretation: during NGRAM decoding with the overlap scheduler and CUDA graphs, a request's radix-cached
KV gets overwritten; every request that later reuses that prefix inherits the corruption. This matches
SGLang #23392 (closed without a fix; it needed cancellations) — here it reproduces deterministically
without cancellations.

**Real-model check (Qwen3-0.6B real weights, greedy):** NGRAM vs no-spec 17/96 requests diverge, but
no-spec default vs no-spec deterministic already diverge in 21/96, so real-model comparisons cannot
separate this bug from benign numerics. The probe's exact oracle can.

Session 3 cost ~$0.10 (35 min at $0.18/h).

## GPU session 2 (RTX 3090, ~$0.29) — every program on both engines
- vLLM 0.31.0 (18 configurations, 557K generated tokens): recall with scale check (Qwen3-0.6B twin),
  posrecall at rope_theta 1e6 (FLASH_ATTN / FLASHINFER × default / chunked 512 + sampling / eager +
  preemption; FlashInfer FP8 KV; n-gram spec), MoE (Qwen3-30B-A3B twin, 4 layers, 128 experts top-8),
  MLA (DeepSeek-V2-Lite twin × TRITON_MLA default / chunked + sampling / eager + preemption / n-gram;
  DeepSeek-V3 twin): **0 mismatches**.
- SGLang 0.5.21 (8 configurations, 262K tokens): recall-sc, posrecall, MoE, MLA on flashinfer and triton:
  **0 mismatches**.

## GPU session 4 (RTX 3090, ~$0.07) — replays and a new MLA observation
- vLLM #42642 and #47728 replays (0.25.1 vs 0.26.0) did not trigger with the configured workloads
  (shared-prefix suffixes too short; async scheduling not enabled) — retried in session 7.
- **MLA probe, DeepSeek-V2-Lite twin, default workload with prefix caching (96 requests, max_model_len
  4084): 7/96 requests wrong from their first token on both vLLM 0.25.1 and 0.26.0** (identical
  requests; not shared-prefix variants). vLLM 0.31 is clean on its own matrix; session 7 reruns the
  identical workload on 0.31 with config variants and bisects 0.27–0.30.
- Localization diagnostics on vLLM 0.31: corruption lands (8/8) and reruns hit the cache; misses come
  from duplicate physical copies of shared blocks (now all copies are corrupted).

## GPU session 7 (RTX 3090) — localization and MLA isolation
- **Targeted KV faults through vLLM 0.31's real paged cache: 40/40 detected, 40/40 localized** (block
  duplication, zeroed block, cross-request block, wrong-layer read, one stale layer; 8 trials each). The
  probe's self-describing cache maps physical blocks to request positions; every physical copy of the
  victim is corrupted, and the first wrong token's reads name the corrupted block.
- MLA (DeepSeek-V2-Lite twin) on vLLM 0.31 with the workload that failed on 0.25.1/0.26.0: exact, also with
  prefix caching off, eager mode, small/large token budgets and chunked prefill off.
- Positional probe v2 without the identical-token prefix fails on GPU (19/96 requests on vLLM 0.31,
  RTX 3090; also on H100): a probe bug (reads before position 0 compound through the accumulator),
  reproduced on CPU at full depth and fixed by the prefix; GPU re-validation pending.

## GPU session 5 (2× RTX 3090, ~$0.12) — tensor and expert parallelism
- vLLM 0.31: MoE twin (Qwen3-30B-A3B shape) with EP=2+TP=2, TP=2, EP=2+EPLB; dense recall + scale check
  with TP=2: **0 mismatches** (98K tokens).
- SGLang 0.5.21: MoE twin with EP=2 + EPLB (16 redundant experts), EP=2+TP=2, TP=2; dense TP=2: **0
  mismatches**.
- SGLang #32962 (EPLB replicas chosen differently per rank) did not trigger on 0.5.16 with these flags.

## GPU session 6 (H100, ~$2.01) — Hopper kernels
- vLLM 0.31 (CUDA 13, torch 2.13): recall + scale check on FA3 (default, chunked + sampling), FlashInfer
  SM90, Triton, FlashInfer FP8 KV, **Triton FP8 KV (supported on Hopper)**, n-gram spec; MoE twin ×2;
  MLA (DeepSeek-V3 twin) on FlashMLA (default, chunked + sampling, FP8 KV), FLASH_ATTN_MLA, TRITON_MLA;
  DeepSeek-V3.2 DSA twin (FlashMLA sparse) with prefix caching and shared prefixes: **0 mismatches**.
- vLLM #36616 replay (V3.2 sparse top-k padding; 0.23.0 vs 0.24.0, prompts up to 3,000 and 8,000): exact on
  both; not triggered (the image lacked `deep_gemm`, so the indexer may not run the affected kernel).
- SGLang 0.5.21: fa3 × {default, chunked + sampling, page 16}, MLA on flashmla and fa3, MoE on fa3: **0
  mismatches**. NGRAM baseline: 6 requests wrong (see the NGRAM table above).
- Positional probe v2 entries failed with the pre-fix probe (no identical-token prefix); re-run in session 10.

## GPU session 7 (RTX 3090, ~$0.19) — MLA bisection and replays
- **Release bisection of the MLA failure** (DeepSeek-V2-Lite twin, replay workload `default/prefix`, 96
  requests, max_len 3,700): 0.25.1, 0.26.0 and 0.27.1 return the same 7 wrong requests (9, 18, 25, 44,
  65, 71, 81; all wrong from their first generated token, 1,792 tokens); **0.28.0, 0.29.0, 0.30.0 and
  0.31.0 are exact**. The 0.27.1 → 0.28.0 range (601 commits) contains three MLA cached-context changes:
  #50613 "Per-request scheduling for MLA chunked context" (its PR reports DeepSeek-V2-Lite accuracy gains
  with prefix caching), #51739 "Optimize long-context MLA cache gathers" and #51733 "Fix MLA prefill
  workspace allocation size". Per-commit wheels pin it in session 9b.
- **vLLM #42642 replayed** ("MLA prefill merge reads the suffix output with the prefix head stride"; fixed
  in 0.26.0): the long-suffix shared-prefix workload (33 requests sharing a cached prefix, suffixes of
  600–1,200 tokens) returns **9 wrong requests on 0.25.1 (2,304 tokens, all wrong from their first
  token) and 0 on 0.26.0** — the bug's fix window exactly.
- #47728 replay: async scheduling with the CPU n-gram drafter is rejected by 0.25.1/0.26.0 ("only
  supported with ... NGram GPU"); retried with `ngram_gpu` and decoy prompts in session 9b.
- Localization retest on 0.31: 40/40 detected and localized (see above); the positional probe entries
  used the pre-fix probe.

## GPU session 8 (RTX 3090, ~$0.04) — bad host
- Every vLLM configuration hit CUDA out-of-memory on its first allocation: `nvidia-smi` showed 24,014 of
  24,576 MiB in use with no process visible in the container. Aborted after 10 minutes;
  `tools/vast_run.sh` now rejects hosts with more than 2 GB of GPU memory in use before the job starts.

## CPU work after session 8
- MLA scale check (above), including DSA.
- **Decoy workload for speculative decoding.** Probe outputs rarely repeat, so n-gram drafters seldom
  propose anything and drafts are almost never rejected. `decoy_variant` plants fragments of the
  prompt's own expected output over filler runs: half continue correctly (accepted drafts), half
  diverge (rejected drafts). Echo tokens are never read, so the oracle is unchanged (CPU: exact on recall
  and MLA twins). New matrix entries: `*/ngram_spec_decoys` (vLLM, SGLang), `default/ngram_gpu_async_decoys`
  (vLLM replay: GPU drafter + async scheduling).
- The CPU fault campaign picks victim blocks the corrupted decode actually reads (a block nobody reads
  cannot change the output); before, a random class-token block was sometimes never read in the window.
- Tests: 52 pass.

## GPU session 10 (H100, ~$0.99) — MLA scale check, DSA, positional probe v2
- vLLM 0.31: MLA (DeepSeek-V3 twin) with the scale check on FlashMLA (default, chunked + sampling),
  FLASH_ATTN_MLA, TRITON_MLA, FlashMLA + n-gram decoys; DeepSeek-V3.2 DSA twin with the scale check
  (prefix caching; long-suffix shared prefixes); **positional probe v2 on FA3, FlashInfer, Triton, FA3
  chunked + sampling: exact** (the identical-token prefix fix holds on GPU); recall + n-gram decoys: **0
  mismatches**.
- SGLang 0.5.21: MLA on flashmla and fa3, positional probe v2 on fa3 (default, chunked + sampling),
  NGRAM with decoys (recall twin): **0 mismatches**.
- **FlashMLA FP8 KV: the scale check fired on all 64 requests** (the main lookups stayed exact). Cause:
  FlashMLA's FP8 path quantizes the absorbed query W_UKᵀq to FP8, and the first MLA construction carried
  the calibrating difference (≈1.3) on top of a ≈500 shared term in the same latent dims, where 3
  mantissa bits erase it. Fixed by giving the A-versus-B difference its own latent dim; a CPU simulation
  (absorbed query and P in FP8, static and dynamic scales) gives CAL 0.620 in the 0.5–0.7 window; new
  CPU tests round dense Q/K/V and the MLA absorbed query to FP8.

## GPU session 11 (H100) — FP8 paths
- vLLM 0.31: **FlashMLA FP8 KV with the fixed scale check: exact**; FlashMLA default and n-gram decoys
  (2,608 drafts, 10,432 draft tokens, 7,597 accepted); DeepSeek-V3.2 with the FP8 sparse cache and with
  prefix caching; **FA3 FP8 KV (query quantized too)** with recall + scale check; n-gram decoys on FA3
  (4,204 drafts, 16,816 draft tokens, 11,798 accepted); positional probe v2 on FA3 FP8 and FlashInfer
  FP8: **0 mismatches**.
- SGLang 0.5.21 FlashMLA FP8 KV: out of memory allocating a 37 GiB workspace at mem_fraction_static 0.8
  (not a correctness result; the matrix now uses 0.6 for that entry).

## GPU session 9b (RTX 3090) — the MLA bug pinned to one commit
- Per-commit wheels (`wheels.vllm.ai/<sha>`), same twin and workload as the release bisection:
  **e7b8d5946 (the parent of #50613): the same 7 requests wrong; b38e111d3 = vLLM #50613 "[Attention][MLA]
  Per-request scheduling for MLA chunked context": exact.** The MLA prefix-caching failure in 0.25.1 –
  0.27.1 is fixed by #50613 (merged 2026-08-06, first released in 0.28.0).
- #47728 replay (Mistral-7B-v0.1 twin, sliding window 4096): with the GPU n-gram drafter, async
  scheduling and decoy prompts, 0.25.1 is exact. The bug corrupts the oldest blocks of the window after a
  rejected draft; the probe keeps every read at least 512 tokens inside the window (engines may drop
  tokens outside it), so those blocks only carry near-zero attention weight. A window-edge reader (a
  positional head with an offset near the window size) would be needed.

## GPU session 12 (RTX 3090) — is the MLA multi-chunk context path still wrong?
#50613 avoids splitting a batch's cached context into lockstep chunks when it fits the prefill
workspace, so the old failure might survive where chunking is still needed. Four workloads with more
cached context per step (prompts up to 12K tokens, 65–128 requests, token budgets up to 65,536):

| Workload | vLLM 0.31 | vLLM 0.27.1 (control) |
|---|---|---|
| prefix caching, 128 requests | 0 | 13 requests wrong (3,328 tokens) |
| token budget 65,536, max 256 sequences | 0 | 2 requests wrong |
| long-suffix shared prefix, 65 requests | 0 | 0 |
| shared long prefix, token budget 65,536 | 0 | 0 |

The workloads still trigger the old failure, and the current release is exact on all of them.

## GPU session 9a (RTX 3090, ~$0.17) — positional probe v2, MLA scale check, decoys, repro script
- vLLM 0.31: **positional probe v2 on 8 configurations** (FA2 default / chunked + sampling / eager +
  preemption, FlashInfer default / chunked + sampling / FP8 KV, Triton, n-gram spec); recall + n-gram
  decoys; MLA with the FP8-robust scale check on the DeepSeek-V2-Lite twin (TRITON_MLA default / chunked +
  sampling / eager + preemption / n-gram decoys) and the V3 twin (default, chunked + sampling): **0
  mismatches** (15 entries).
- SGLang 0.5.21: positional probe v2 on flashinfer, triton, flashinfer chunked + sampling; NGRAM with
  decoys; MLA V2-Lite twin on triton and flashinfer: **0 mismatches**. The V3 twin ran out of memory on
  the 24 GB card.
- `tools/repro_sglang_ngram.py` end to end on a fresh machine: "NGRAM vs no-spec: 8/192 requests differ;
  NGRAM vs exact oracle: 8 wrong" — the maintainer-facing repro works.
- Tooling lesson: editing `tools/vast_run.sh` in place while a session ran made bash (which reads scripts
  incrementally) re-run part of it, so the job started a second time; the results were pulled by hand,
  the duplicate entry dropped, and the runner now keeps its whole body in one function.

## B200 attempts (session 13)
Two Oregon B200 hosts parked the container as "stopped" and never started it (no GPU charge); the runner
now gives up on stopped instances after 3 minutes. A third host (Virginia) started normally.

## GPU session 13 (B200, ~$2.97) — first Blackwell coverage
- vLLM 0.31 on SM100: recall + scale check on FLASH_ATTN and FLASHINFER (TRT-LLM-gen kernels), FlashInfer
  FP8 KV, n-gram decoys (the same 4,204 drafts / 11,798 accepted tokens as on the H100: exact outputs make
  drafting identical across correct engines); positional probe v2 on FlashInfer; MLA V3 + scale check on
  CUTLASS_MLA (default and FP8 KV), FLASHINFER_MLA, TRITON_MLA; MLA prefix caching and split-marker prompts:
  **0 mismatches**.
- vLLM #52796 replay (FlashInfer / TRT-LLM-ragged MLA prefill return base-2 LSE that vLLM merged as a
  natural log after a prefix hit): 0.28.0 and 0.29.0 both exact with split markers on the DeepSeek-V3
  twin. Reason: 0.28.0 prefers the TRT-LLM ragged prefill only for MLA head dims 192/64/256 (GLM-5
  style, the model in the original report); DeepSeek-V3 dims get FlashAttention prefill. Session 14
  reruns it with a 192/64/256 twin and with the prefill backend forced.

## GPU session 14 (B200, ~$2.04) — vLLM #52796 re-caught
FlashInfer and TRT-LLM-ragged MLA prefill kernels return base-2 log-sum-exp values; vLLM 0.28.0 merged
them as natural logs when combining cached-context and suffix attention after a prefix hit. The main
lookups are immune (30+ nat margins); the scale check is not, once its two markers sit on opposite sides
of the cached prefix (split-marker prompts). 64 requests per entry:

| | 0.28.0 | 0.29.0 | 0.31.0 |
|---|---|---|---|
| GLM-style MLA head dims 192/64/256 (TRT-LLM ragged prefill by default) | 62 wrong | 0 | 0 |
| DeepSeek-V3 twin, `mla_prefill_backend=TRTLLM_RAGGED` | 62 wrong | 0 | 0 |
| DeepSeek-V3 twin, `mla_prefill_backend=FLASHINFER` | 59 wrong | 0 | 0 |

Every wrong request emits the scale-check token at its first step ("softmax scale"); the base prompt
(no prefix hit) is exact. vLLM's own log confirms the prefill backend in each run.

## M4 — release (2026-10-07)
- Docs: README (results, CI recipe), `docs/DESIGN.md`, `docs/RESULTS.md`, `docs/REPLAY_CANDIDATES.md`,
  `tools/README.md`; Apache-2.0; GitHub Actions CI (CPU tests on Python 3.11 and 3.12).
- Clean-install check: the release snapshot installed into a fresh Python 3.10 venv (torch 2.14.1 CPU,
  transformers 5.19.0) passes all 57 tests in 77 s.
- Totals: 207 GPU matrix entries, about 5.8M generated tokens, $13.08 of GPU credit; every mismatching
  entry is explained (`python tools/summarize_runs.py`).
- Publishing: the `release` branch holds the snapshot as one commit authored with the GitHub noreply
  address (local `main` keeps the full history). Creating the public repository was blocked by the
  session's permission policy, so it is left to the maintainer:
  `gh repo create DiaaAj/probec --public --source . --remote origin && git push -u origin release:main`.
- Left for the maintainer on purpose: filing the SGLang NGRAM issue and sharing the vLLM MLA note (drafts
  in the untracked `notes/`), and a PyPI release (needs credentials).

## GPU sessions 15 / 15b (RTX 3090, ~$0.15) — the SGLang NGRAM bug on SGLang main
No SGLang release followed 0.5.21, so main was pinned at d0a098b (2026-10-12, 0.5.22.dev813).
- 0.5.21 control on a third machine: 8/192 wrong — the same requests and steps as on 2026-10-07.
- main: the first attempt could not build (the image has no Rust toolchain; main's radix tree is a Rust
  crate, so building without the extensions would test different code). Built with Rust: **0/192 wrong**.
- NGRAM's argument handling is unchanged on main (`_handle_ngram` differs only in a DP-attention check;
  overlap is disabled only on CPU in both), so the overlap scheduler, radix cache and CUDA graphs were all
  active. The change comes from main's reworked KV-cache / memory-pool code (Rust radix tree,
  unified-memory pool for speculative decoding, radix lock refactors) — fixed there, or no longer
  triggered by this workload; a bisection over main would tell which.
- The issue draft now reads: affects the current release 0.5.21; appears fixed on main.

## GPU session 16 (RTX 3090, ~$0.19) — the SGLang NGRAM bug on 0.5.22
SGLang 0.5.22 was released on 2026-10-16; its tag contains main @ d0a098b plus 139 newer commits.
- 0.5.22 (pip wheels; torch 2.14.1, sglang-kernel 0.4.9, FlashInfer 0.7.1): **NGRAM repro 0/192 wrong**
  (NGRAM vs no-spec 0/192).
- 0.5.21 control on the same machine (a fourth host, in Sweden): 8/192 wrong, the same requests and first
  divergent steps as on 2026-10-07 and 2026-10-12.
- So the failure is gone in the current release. Whether 0.5.22 fixed it or only stopped this workload
  from triggering it is still open; a bisection over main (v0.5.21 → d0a098b, about 813 commits) would
  tell. Nobody else has reported it on SGLang's tracker.
- Default SGLang matrix on 0.5.22: flashinfer and triton × {default, chunked 512 + sampling, no radix +
  eager + memory pressure, page 16} exact (8 entries, 393K tokens). flex_attention then failed with an
  Inductor lowering error (decode tile sizes do not divide the 128-token sparse blocks on SM86), and the
  failing scheduler sent SIGQUIT to its parent, which ended the whole `probec run` process: the FP8 KV and
  NGRAM entries were never reached. Harness fix needed: run each SGLang entry in its own process (the M1
  job did this in shell with `--only`).
- First attempt: a Czech host parked the container as stopped; destroyed after 3 minutes. The Swedish
  host took 17 minutes to pull the image.

## GPU session 17 (RTX 3090, ~$1.00) — bisecting the SGLang NGRAM bug
v0.5.21 is a release-branch tag two cherry-picks (a test change and an InternVL fix) past main's bd78095,
so the bisection ran on main from bd78095 to d0a098b: 813 commits. Each test builds SGLang from source in
a fresh venv (uv; `git clean -ffdx` first; Rust extensions included, with a shared Cargo target dir) and
runs `tools/repro_sglang_ngram.py`, which now also prints the run without speculation against the oracle
and the wrong request ids. A commit that failed to build, or whose run without speculation was wrong,
would have been skipped; none was.
- Both ends re-tested first on this machine: bd78095 8/192 wrong (the same 8 requests as the 0.5.21
  wheels), d0a098b 0/192.
- 10 steps. **First clean commit: e14ef4b = #42720, "[Scheduler] Skip the decode for a request whose last
  token is already in flight".** Its parent 1fc6b89 is a docs-only commit with the same 8 wrong requests;
  both were tested again at the end, with the same results. Every failing commit failed the same 8
  requests. The dependency upgrades (torch 2.14.1, sglang-kernel 0.4.9, FlashInfer 0.7.0.post1) all land
  before position 406, which still fails.
- #42720 is an optimization; its test counts decode passes. In overlap mode it drops the trailing decode
  whose output was thrown away, but only for requests that finish by `max_new_tokens`; its description
  says requests that end on EOS or a stop string still run it. Every request in our workload ends by
  length, so the bug is most likely hidden in 0.5.22, not fixed. Next: 0.5.22 with requests that end on
  stop tokens, and 0.5.22 with the skip switched off.
- uv used the image's managed Python 3.11 (`/opt/uv/python`), not the system 3.12; the bug reproduces
  the same under it. Source builds report version "0.5.22" (SGLang computes it with
  `scripts/release/get_version_tag.py`), so the step script's `+g<sha>` check had nothing to check; uv's
  log shows each build came from the checkout.
- A step took about 10.5 minutes: about 6.3 to build (about 3.3 of them recompiling the Rust workspace
  crates even with a warm target dir) and 3.8 to run. The 120-minute step budget would have stopped the
  bisection after 8 steps, so the deadline was extended on the box (the step script is re-read by each
  step).
- First attempt: a Utah host parked the container as stopped; destroyed after 3 minutes. The Czech host
  pulled the image in about 2 minutes.

## M5, step 1 — SGLang matrix entries in their own processes (2026-10-18, CPU)
Session 16's matrix ended at its first crashing entry: when a scheduler fails, SGLang sends SIGQUIT to
its parent, whose handler kills the whole process tree, and the parent was the `probec run` process.
- `probec run --engine sglang` without `--only` now runs each entry as `probec run ... --only <entry>`
  in a child process with its own session, so killing the child's tree cannot reach the parent. A child
  that exits, is killed or exceeds `--entry-timeout` (default 1800 s) without writing a record gets an
  error record with the last 20 lines of its output; the run then continues. After each entry the
  child's process group is killed, so leftover engine processes cannot hold GPU memory. A parent stopped
  by SIGTERM (a job's `timeout`) stops its child first. `--in-process` keeps the old behaviour; vLLM
  entries still run in one process.
- Tests: fake entries that write a record, exit with an error, kill their own process tree (as SGLang
  does) and hang; and the SGLang route builds one `--only` child per matrix entry. 59 CPU tests pass. A
  local run of the `quick` matrix without SGLang installed produced one error record per entry from real
  child processes.
- Still to validate on a GPU: the default SGLang matrix on 0.5.22 should now reach the FP8 KV and NGRAM
  entries after flex_attention fails.

## GPU session 18 (RTX 3090, ~$0.30) — the SGLang NGRAM bug is still in 0.5.22
Run by a background agent on the Czech host from session 17 (one launch, 30 minutes). The repro gained
`--finish stop`: each request ends on its own stop token (an oracle token at a random position 8–240 that
does not occur earlier; `ignore_eos` off), and the oracle answer is cut there. pip wheels, Python 3.12,
as in session 16.
- 0.5.22 with length endings: 0/192. With stop-token endings: **3/192 wrong** (58, 109, 185), the run
  without speculation exact; 0/192 with the overlap scheduler off.
- 0.5.22 with only #42720's skip switched off (`self.enable_skip_finishing_decode = True` → `False` in
  `event_loop_overlap`, one line): **8/192, the same requests and first divergent steps as 0.5.21.**
- 0.5.21: 8/192 with length endings (control), 3/192 with stop-token endings, the same 3 requests as
  0.5.22 with identical NGRAM verify counts.
- 0.5.22 with forced retraction (`SGLANG_TEST_RETRACT=1`), length endings: 0/192.
- So the defect is still in 0.5.22, and #42720 hides it only for requests that end by length. EOS
  endings go through the same check as stop tokens (`_check_token_based_finish`), so EOS-terminated
  traffic is most likely affected too; not measured.
- As before, a request owning a cached prefix goes wrong mid-generation (58 at token 137) and requests
  reusing cached prefixes are wrong from their first reads (109, 185 at token 4).
- The agent's local dry run compiled the full 28-layer twin on the laptop while my activation analysis
  was running; together they froze it (swap nearly full, load about 136). Both have been stopped, and heavy
  work now goes to rented boxes.

## M6 — quantized weights (from 2026-10-18)
Design: `docs/QUANTIZED.md`. Code: `src/probec/quant.py` (exact weight quantizers, activation fake-quant,
calibration) and `src/probec/quantize.py` (`probec quantize`: FP8 per tensor and in 128×128 blocks, AWQ,
GPTQ, NVFP4 checkpoints in the layouts vLLM 0.31 and SGLang 0.5.22 load, read from both engines' source by
a research agent). CPU tests decode every format with the engines' formulas and get the designed weights
back bit for bit.

### GPU session 19 (RTX 3090, ~$0.22): online FP8, the full twin fake-quantized, the harness fix on a GPU
- `probec certify --quant all` on the Qwen3-0.6B twin (no scale check): all 196 quantized linears exact in
  FP8 (per tensor, 128-blocks), INT4 (AWQ, GPTQ), NVFP4 and MXFP4.
- The reference model (HF transformers, on the GPU) with FP8 weights and fake-quantized activations, 24
  prompts up to 4,000 tokens, 64 new tokens each: exact with FP8 per-tensor, per-token and 1×128-group
  activations (0/1,536 each). The NVFP4 and MXFP4 runs crashed on a device mismatch in quant.py (fixed;
  rerun in session 20).
- vLLM 0.31 online FP8 (FP8 Marlin, weight-only, on SM86): 4 entries exact (196,608 tokens). Two entries
  could not start: the engines torn down before them in the same process still held about 4 GB, so vLLM
  entries now run in their own processes too.
- SGLang 0.5.22 `--matrix default`, every entry in its own process: 11 entries exact (540,672 tokens).
  flex_attention ×3 and Triton FP8 KV (unsupported on SM86) crashed and got error records, and the run went
  on: the FP8 KV and NGRAM entries that session 16 never reached ran exact. **The harness fix works on a
  GPU.**
- SGLang online FP8 (Marlin W8A16): 5 entries exact (245,760 tokens).

### GPU session 20 (H100 PCIe, ~$3.53): quantized checkpoints on Hopper
Twins (no scale check): Qwen3-0.6B, Qwen3-8B-shaped (4 layers), Qwen3-30B-A3B-shaped MoE (4 layers, 128
experts), DeepSeek-V3 MLA (3 layers). `probec quantize` wrote FP8 per-tensor and 128×128-block, AWQ,
GPTQ and NVFP4 copies; the MoE AWQ copy failed (an INT4 zero-point bug: groups without zeros, fixed). One
entry per checkpoint and engine.
- The reference model with NVFP4 and MXFP4 activations (the session 19 rerun): exact (0/1,536 each). So
  the full twin is exact under all five activation schemes; the off-grid values of docs/QUANTIZED.md §1
  never reach a meaningful token.
- **vLLM 0.31, all exact:** online FP8 W8A8 (CUTLASS, per-token activations; 6 entries); probec's FP8,
  FP8-block (FlashInfer + DeepGEMM, scales requantized to UE8M0), 8B FP8-block, MoE FP8-block (Triton
  fused MoE), AWQ and GPTQ (Machete), 8B AWQ, NVFP4 (Marlin W4A16), DeepSeek-V3 MLA FP8-block on FlashMLA
  and Triton MLA.
- **SGLang 0.5.22:** exact on online FP8 (5 entries), FP8, FP8-block (DeepGEMM), MoE FP8-block, AWQ, GPTQ,
  8B AWQ and MLA FP8-block. **NVFP4: 192/192 requests wrong from the first token** (49,133 of 49,152
  tokens), the same count as on the RTX 3090 in session 21.

### GPU session 21 (RTX 3090, ~$0.22): quantized checkpoints on Ampere (all Marlin)
Dense Qwen3-0.6B and a 2-layer Qwen3-30B-A3B-shaped MoE twin; MoE AWQ (with the INT4 fix) and MoE NVFP4
(per-expert calibration) are new. Certify: every format exact on both twins, except MXFP4 on the MoE twin
(520/776: power-of-two scales cannot hold its 15/32 expert weights; no engine runs MXFP4 there).
- **vLLM 0.31: all 10 checkpoints exact** (FP8, FP8-block, AWQ, GPTQ, NVFP4; dense and MoE; Marlin W8A16,
  W4A16, Marlin MoE).
- **SGLang 0.5.22:** exact on FP8, AWQ and GPTQ. **NVFP4, dense and MoE: every request wrong** (49,133 and
  49,135 tokens). **FP8-block (dense) and AWQ (MoE): one request wrong each** (request 65, 245 tokens from
  step 11 of a 5,876-token prompt; request 16, 111 tokens from step 145), while vLLM was exact on both;
  session 23 investigates. MoE FP8 and FP8-block: Triton `fp8e4nv` is not supported on SM86 and SGLang has
  no Marlin fallback for FP8 MoE (a crash, as the format research predicted); MoE GPTQ: `moe_wna16_marlin_gemm`
  asserts fp16 scales with a bf16 model (a crash). Those crashes ran under `--only`, which still ran in
  place: SGLang's SIGQUIT killed `probec run` before it wrote error records; `--only` is now isolated too.
- **The NVFP4 failure is SGLang's copy of a bug vLLM fixed in #34577** ("Rescale NVFP4 weight scales to
  fix BF16 dequant underflow", 2026-03): Marlin turns E4M3 block scales into an S0E5M3 form with
  `(scale * 2**7).view(int16) << 1`, which assumes `scale * 2**7 >= 2`. probec's NVFP4 uses
  `weight_scale_2 = 1` with exact power-of-two block scales as small as 2^-7, so they underflow. SGLang
  0.5.22 and main (2026-10-18) still have the unfixed code; it is SGLang's NVFP4 path on every GPU below
  Blackwell. `probec quantize --nvfp4-scales modelopt` writes ModelOpt-style scales as the control.

### GPU session 22 (B200, ~$7.09): quantized checkpoints on Blackwell
- **vLLM 0.31, all 15 entries exact:** online FP8 W8A8 (6); probec's FP8, FP8-block (DeepGEMM with UE8M0
  scales), AWQ (Marlin), **NVFP4 W4A4** with probec's unit scales and with ModelOpt-style scales (FlashInfer
  / CUTLASS FP4; 4-bit weights and activations, the calibrated static `input_scale`), MoE FP8-block and
  **MoE NVFP4 (TRT-LLM FP4 MoE)**, DeepSeek-V3 MLA FP8-block on CUTLASS MLA and FlashInfer MLA.
- **SGLang 0.5.22:** exact on online FP8 (5), FP8, FP8-block, AWQ, NVFP4 (both scale styles), MoE FP8-block
  and MoE NVFP4. So the unit-scale NVFP4 checkpoints that SGLang garbles below Blackwell are exact on its
  W4A4 path: the fault is in its Marlin path, not the checkpoint. MLA FP8-block on the Triton backend ran out
  of memory (a 256 GiB allocation during setup on a 178 GB GPU; loud, not wrong output), and this time the
  crash left an error record (`--only` now isolated).

### GPU session 23 (RTX 3090, ~$0.15, aborted)
The Oklahoma host downloaded at about 680 KB/s: SGLang's pip install ran 40 minutes without finishing, so the
instance was destroyed and the job reran as session 23b on the Czech host.

### GPU session 24 (RTX 3090, ~$0.18): two quantized-path bugs re-caught on their last buggy release
| vLLM, RTX 3090 | last buggy | fix release |
|---|---|---|
| **#44735** FP8 Marlin skips the transpose of square layers: Qwen3-8B-shaped twin (o_proj 4096 × 4096), probec FP8 checkpoint | 0.23.0: **192/192 wrong** (49,094 tokens) | 0.24.0: exact |
| the same twin, online `quantization="fp8"` | 0.23.0: **192/192 wrong** (49,112) | 0.24.0: exact |
| Qwen3-0.6B-shaped twin (no square layer), FP8 checkpoint: control | 0.23.0: exact | 0.24.0: exact |
| **#34577** NVFP4 Marlin scale underflow: dense twin, probec's unit scales | 0.18.1: **49,133 wrong** | 0.19.0: exact |
| the same weights, ModelOpt-style scales: control | 0.18.1: exact | 0.19.0: exact |
| MoE twin, unit scales | 0.18.1: **49,135 wrong** | 0.19.0: exact |

Every run selected FP8 Marlin (`MarlinFP8ScaledMMLinearKernel`) or NVFP4 Marlin, as the bugs need. **The
NVFP4 counts are identical to SGLang 0.5.22's on the same checkpoints** (49,133 dense, 49,135 MoE): SGLang's
NVFP4 Marlin path behaves exactly like vLLM before #34577.

### GPU session 23b (RTX 3090 CZ, ~$0.42): SGLang's quantized-path failures on Ampere
Rows are runs of one matrix entry's workload (192 requests × 256 tokens, `tools/sgl_variant.py`), with one
setting changed. Some rows were appended while the job ran; bash reads a script as it executes it.
- **NVFP4:** probec's unit scales 49,133 tokens wrong, as in sessions 20/21; the same weights with ModelOpt-style
  scales exact. vLLM 0.31 exact on both. The cause is the scale underflow (vLLM #34577 not ported).
- **Dense FP8-block (FP8 Marlin, W8A16):** default runs wrong in 5 of 7 (1–5 requests each, including session
  21 on another host), with Triton attention too; radix cache off 1 of 5, CUDA graphs off 1 of 3, **overlap
  scheduler off 0 of 5**, the bf16 control 0 of 3. A failing request's lookups find nothing: from then on it
  emits the probe's default token. Requests that share a prefix fail together. `tools/marlin_determinism.py`
  checked SGLang's FP8 Marlin GEMM against the exact product: every M from 1 to 1024 plus larger sizes (10,330
  calls), 20 repeats per size, two streams and a side stream all exact. The engine's use of it is at fault, not
  the kernel; not located yet.
- **MoE AWQ (Marlin MoE):** wrong in 7 of 8 runs: default 5 and 17 requests, radix cache off 1, overlap off 2,
  CUDA graphs off 1, half the KV memory 18 then 0, **one request at a time 11**. The MoE program's diagnostic
  tokens name the experts with wrong outputs (64, 65, 70, 75, 92 in one run). **Exact with the Triton WNA16 MoE
  kernel (`quantization="moe_wna16"`, 2 of 2) and on vLLM 0.31 (2 of 2)**; the bf16 MoE control exact at 0.8 and
  0.9 memory fractions. A KV-index overflow is ruled out (it fails with half the pool).
- `probec run --only` on a crashing entry (SGLang MoE FP8 on SM86) now leaves an error record.
- Issue drafts (not filed, `notes/`): Marlin MoE, FP8 Marlin with the overlap scheduler.

## M7 — hybrid linear-attention models (from 2026-10-18)
Design: `docs/HYBRID.md`. Targets Qwen3.5 (text models `Qwen3_5ForCausalLM`; both engines register them),
whose Gated DeltaNet layers make up 3 of every 4. Each GDN layer is a hop of the recall chain: its recurrent
state is a table with one slot per class (16 key heads × 128 key dims = 2,048 slots), filled by saturated
detector banks. With β = 1 and no decay the delta rule overwrites slots exactly. Echo tokens write a skip
value into their class's slot, so decode-time writes, and rejected draft tokens that an engine fails to
roll back, change later hops.

### CPU (HF transformers 5.19, torch fallbacks of the FLA and causal-conv kernels), 2026-10-18
- Built-in shapes `tiny-qwen3.5` (Qwen3.5-0.8B widths) and `tiny-qwen3.5-gva` (two value heads per key head),
  plus twins from the real 0.8B / 4B / 27B configs (the multimodal config's text model; vision tower and MTP
  head dropped).
- `probec check --quick` on the 4-layer twin: **0 mismatches** (eager and SDPA, fp32 and bf16, left-padded
  batches, sampling), **36/36 layer states bit-identical**, chunked prefill and recurrent decode.
- State faults: zeroed, stale (an earlier block boundary's state), cross-request and wrong-layer states
  detected in every trial at the first decode step; three random rejected drafts left in the state detected
  in 1 of 2 trials (only when the walk later passes their classes; a draft on the next step's chain is always
  caught); zeroed conv states not detected (v1 reads only the current conv tap, as documented).
- 63 of 192 GDN hops in 64 decode steps read a slot last written by an echo; decoys keep the oracle.
- `tests/test_hybrid.py`: 19 tests (both shapes at 8 layers, 2 of them attention layers).

### GPU session 25 (RTX 3090 US, ~$0.08): CPU reference on the box, first vLLM entries, results lost
- On the box (HF transformers 5.17, 72 CPUs): the full-depth Qwen3.5-0.8B twin (24 layers, 18 of them GDN) had
  0 mismatches and 276/276 bit-identical layer states; state faults were 2/2 for zeroed, stale, cross-request,
  wrong-layer and rejected drafts, and 0/2 for a zeroed conv state. The 4B twin (8 layers, two value heads per
  key head) had 84/84, with drafts 1/2. All four twins certified (codes in unrotated dims included).
- vLLM 0.31 with default settings (prefix caching on, which caches states in "align" mode; CUDA graphs) got
  8, 17 and 10 of 192 requests wrong on FLASH_ATTN, FLASHINFER and TRITON_ATTN. Each failure starts mid-decode
  (steps 2–61, never the first token). Two requests with the same expected output but prompts of different
  lengths fail at the same step. After the first error the outputs converge to the sequence the probe gives
  when GDN reads come back empty, as if a request ran on another request's state. Both shared-prefix workloads
  were exact; there every request has the same pool, so a swapped state would not show.
- After 30 minutes the host closed the ssh session that ran the job, the final pull failed, and the runner
  destroyed the instance; only the entries above survived, through a monitor. The runner now runs jobs
  detached and pulls results every 4 minutes. Sessions 25b (vLLM, with the `hybriddebug` matrix) and 25c
  (SGLang) rerun everything.

### Sessions 25b / 25c (RTX 3090, ~$0.21): a probe bug in the hybrid program, found and fixed
- vLLM (25b) and SGLang (25c) both got 8–22 of 192 requests wrong on every fresh-prompt entry of the hybrid
  matrix. The entry with seed 0 failed **the same requests at the same steps with the same wrong tokens** on
  both engines. Shared-prefix workloads were exact, except one where every request shares a single base
  prompt, which therefore fails all-or-nothing.
- The reference model (HF, CPU), run on those prompts, also gave the engines' tokens, so the oracle was
  wrong, not the engines. Tracing request 113 layer by layer:
  - Layers 0–18 were bit-exact at the failing step.
  - Layer 19 (attention) read class token 601. That prompt token's residual had an RMS of 13 instead of 8,
    so its payload came out scaled (−0.219 instead of −1).
  - Layer 20 (GDN) then read class 1630's slot with a negative query and got the value with its sign flipped.
- **Cause:** prompt tokens whose chains break in prefill carry junk pointers. The GDN query's L2 norm and the
  gated RMSNorm both divide by the read's own size, so mildly negative detectors read slots at full strength.
  Each layer added more junk, until the residuals of tokens that are read later were off by 60%.
- **Fix:** each GDN key head keeps a null key dim that nothing writes and every query reads with 65,536. A
  stray detector then reads at about 1% of the gated norm's floor, and a match reads 2,000 times above it.
  The detectors are sharper too: 1024·(dot − 62). Hybrid twins hold 2,032 classes, one null slot per head.
  An intermediate fix (slope 16, no null slot) moved the band where reads flip instead of removing it: drift
  2.1·10⁻³ on a 12-layer twin.
- **New check:** `probec check` reports `residual_drift`, the largest drift of a read token's RMS from 8 over
  all layers, and fails above 10⁻³. After the fix: 4.0·10⁻⁴ on the full-depth twin (4.2·10⁻⁴ on the 12-layer
  one, the same as the attention-only twins' junk), and the reference model is exact on requests that failed.
- Sessions 26a (vLLM) and 26b (SGLang) rerun the hybrid matrices with the fixed twins.

### Sessions 26a / 26b (RTX 3090, ~$0.53): the fixed hybrid twins on both engines
- **SGLang 0.5.22:** all 14 entries of the hybrid matrix on the full-depth Qwen3.5-0.8B twin exact (688,128
  tokens). That covers NGRAM with decoys (rollback of rejected drafts), memory pressure, a bf16 state, the radix
  strategies, and the overlap scheduler, radix cache and CUDA graphs each off. The 4B, 27B, 35B-A3B MoE and
  Qwen3-Next twins were exact too.
- **vLLM 0.31:** 12 of 15 entries exact. The 3 n-gram entries were wrong in 8, 12 and 20 of 192 requests
  (default; Triton verify kernel; prefix caching off). An oracle replay at each first wrong step explains every
  failure by an accepted token's GDN write missing from the state, mostly the token accepted in the previous
  step; none by a kept rejected draft. A new, unreported vLLM bug (`docs/RESULTS.md` §7.3); session 27 isolates
  it. The other four twins were exact.
- **Reference model on the box (full-depth twin):** exact; residual drift 3.6·10⁻⁴.
  - State faults: 2/2 each for zeroed, stale, cross-request, wrong-layer and rejected drafts; 0/2 for a
    zeroed conv state.
  - KV faults: 2/2 each for block_dup, cross_request, zero_block and wrong_layer; 0/2 for stale_partial.
- **SGLang Marlin MoE (AWQ), one request at a time** (`tools/sgl_repeat.py`): the same request run 24 times
  gave 3–6 distinct outputs, 12 of 96 runs wrong. With the Triton WNA16 kernel, 0 of 32.
- The detached runner worked on both boxes: output streamed, results pulled, exit status seen.

### Session 27b (RTX 3090 SE, ~$0.10; session 27's host never pulled the image): the vLLM n-gram bug isolated
The `hybridspec` matrix reruns 26a's failing entry (seed 12, the same 192 prompts) with one setting changed:
- **Deterministic:** the same 967 wrong tokens in the same 8 requests on a rerun and on another host.
- **No effect:** async scheduling off, `enforce_eager`, and `max_num_seqs=1` (one request at a time).
- **Still fails:** 1 and 2 speculative tokens (1 and 7 requests), and the GPU n-gram drafter (5).
- **Exact:** the default workload, where the drafter proposes only correct continuations (32,712 of 32,712
  draft tokens accepted).

So it needs a partly accepted draft, where the state must be taken at an intermediate position. Issue body:
`notes/filing/vllm_hybrid_ngram.md`.
