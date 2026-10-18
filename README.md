# probec

**Exact known-answer checkpoints for testing LLM inference engines.**

probec ("probe compiler") compiles small programs into the weights of a real model architecture: a
*probe twin* of the model you serve. Every floating-point operation in the probe lands in a regime
where it is exact, so an unmodified engine (vLLM, SGLang, …) running its real kernels must produce
one known token stream, bit for bit. There is no reference model and no tolerance:

- any wrong token points to an engine bug: the construction leaves no room for benign numeric
  differences (validation found and fixed two probe-side gaps on the way; see `docs/RESULTS.md`);
- the wrong token says what went wrong: which KV-cache block was misread, which expert a token was
  sent to, or that the softmax scale is off.

It is a Tracr-style compiler (programs → transformer weights), built for exactness on production
architectures instead of interpretability.

> Status: v0.1.0. Validated on vLLM 0.31 and SGLang 0.5.21 / 0.5.22 on RTX 3090, H100 and B200 GPUs (about
> 11.3M generated tokens checked), including quantized checkpoints in FP8, FP8 blocks, AWQ, GPTQ and NVFP4.
> It caught seven real engine bugs, three of them unreported; see [`docs/RESULTS.md`](docs/RESULTS.md).

## What it catches

| Program | Architectures | Catches |
|---|---|---|
| `recall` | Llama, Mistral, Qwen2, Qwen3 (dense GQA, QK-norm, sliding window) | KV reads from the wrong block, slot, layer or request; stale or never-written blocks; prefix-cache, chunked-prefill, preemption and speculative-decoding KV bookkeeping errors; softmax-scale faults (dropped or doubled 1/√d, exp vs exp2, wrong key scale) via the built-in scale check |
| `posrecall` | same, at the model's real `rope_theta` and llama3 scaling | position-id slips, RoPE frequency or pairing errors, keys cached with the wrong position |
| `moe` | Qwen3-MoE | tokens routed to the wrong expert, dropped or duplicated experts, wrong top-k weights; the output names the expert |
| `mla` | DeepSeek V2/V3 (MLA), V3.2 (DSA sparse attention) | latent-cache misreads, MLA prefill/decode path errors, sparse top-k sets that lose the token being read, softmax-scale faults (FP8-robust) |

Workloads include shared prefixes (short and long suffixes, to reach cached-context prefill paths);
*decoy* prompts that plant fragments of the expected output, so n-gram speculative decoding proposes
drafts that get accepted and rejected; and *split-marker* prompts that put the scale check's two markers
on opposite sides of a cached prefix, so a wrong merge of cached-context and new-token attention shows.

It does **not** see small numeric drifts (accumulation order, slightly wrong scales): the rounding
basins that make it exact also hide them. See [`docs/DESIGN.md`](docs/DESIGN.md).

## Results so far
Details and raw numbers: [`docs/RESULTS.md`](docs/RESULTS.md).

- **Correct paths stay exact:** about 11.3M generated tokens across vLLM 0.31 and SGLang 0.5.21 / 0.5.22
  configurations on RTX 3090, H100 and B200 (attention backends incl. FA3, FlashInfer, Triton, FlashMLA,
  sparse FlashMLA, CUTLASS-MLA; prefix caching, chunked prefill, preemption, CUDA graphs, FP8 KV, n-gram
  speculation, tensor and expert parallelism; quantized weights through Marlin, CUTLASS, DeepGEMM,
  Machete, FlashInfer FP4 and TRT-LLM MoE kernels). Every token that went wrong traced to an engine bug
  (below) or to a probe bug that was found and fixed during validation, except two SGLang failures on
  quantized checkpoints that are still under investigation (`docs/RESULTS.md` §6.1).
- **Faults are caught and located:** deliberately corrupted KV blocks in vLLM's real paged cache were
  caught 40/40 times, and the corrupted block was named 40/40 times.
- **Real bugs:**
  1. *SGLang 0.5.21 and 0.5.22, unreported:* NGRAM speculative decoding corrupts radix-cached KV when
     the overlap scheduler and CUDA graphs are on and requests sharing prefixes run concurrently (8/192
     requests; deterministic: the same requests fail at the same steps on three RTX 3090 hosts, and it
     also shows on an H100). Every request that reuses the corrupted prefix inherits the error.
     Real-model comparisons cannot isolate it; the probe's exact oracle does. A bisection stops at
     #42720, a scheduler optimization that drops a discarded trailing decode, but only for requests that
     end by length. With requests ending on stop tokens, 0.5.22 fails 3/192 like 0.5.21, and switching
     #42720's skip off brings back the 8/192.
  2. *vLLM 0.25.1–0.27.1, unreported:* MLA with prefix caching returns wrong outputs for 7/96 requests
     on a DeepSeek-V2-Lite twin, from their first token. Bisected with per-commit builds to the fix:
     #50613, merged as a prefill-scheduling optimization.
  3. *vLLM #42642, re-caught:* the long-suffix shared-prefix workload breaks 9 of 33 requests on 0.25.1
     and none on 0.26.0, the release with the fix.
  4. *vLLM #52796, re-caught on Blackwell:* MLA prefill kernels returned base-2 log-sum-exp values that
     were merged as natural logs after a prefix hit. Prompts that put the scale check's two markers on
     opposite sides of the cached prefix break 59–62 of 64 requests on 0.28.0 and none on 0.29.0.
  5. *SGLang 0.5.22, unreported:* NVFP4 checkpoints with small block scales come out garbled on every GPU
     below Blackwell, from the first token of every request: the Marlin fallback's scale conversion
     underflows. vLLM fixed the same code in #34577; SGLang does not have the fix yet.
  6. *vLLM #44735 and #34577, re-caught:* FP8 Marlin skipping the weight transpose of square layers
     (0.23.0: 192/192 requests wrong on a Qwen3-8B-shaped twin) and the NVFP4 Marlin scale underflow
     (0.18.1: the same token counts as SGLang 0.5.22's).

## Quick start

```bash
pip install -e ".[dev]"                                  # torch, transformers (CPU is fine)

# a probe twin of a real model (config.json from the Hub or a local path), fewer layers
probec compile --like Qwen/Qwen3-8B --layers 4 -o probes/qwen3-8b
probec compile --like Qwen/Qwen3-30B-A3B --layers 4 --program moe -o probes/qwen3-moe
probec compile --like deepseek-ai/DeepSeek-V3 --layers 3 --program mla -o probes/dsv3
probec compile --like Qwen/Qwen3-8B --program posrecall -o probes/qwen3-8b-pos

probec certify probes/qwen3-8b          # static exactness conditions (margins, energy, dyadic weights)
probec check probes/qwen3-8b --quick    # reference run on CPU with HF transformers, plus fault injection
```

On a GPU, run the probe through an engine across a matrix of serving configurations:

```bash
pip install -e ".[vllm]"     # or ".[sglang]"
probec run probes/qwen3-8b --engine vllm --list            # show the matrix
probec run probes/qwen3-8b --engine vllm --matrix default  # backends × prefix cache × chunked prefill × FP8 KV × spec decode
probec run probes/qwen3-8b --engine sglang --matrix default
probec report out/results.jsonl                            # markdown summary
```

Quantized paths (design: [`docs/QUANTIZED.md`](docs/QUANTIZED.md)); compile these twins with
`--no-scale-check`, whose calibration needs more precision than FP8 or FP4 keep:

```bash
probec certify probes/qwen3-8b --quant all                 # do the weights quantize exactly in each format?
probec check probes/qwen3-8b --quant-act fp8-token --quant-weight fp8 --device cuda   # reference model, fake-quantized
probec run probes/qwen3-8b --engine vllm --matrix quant    # online FP8 in the engine
```

Matrices: `quick`, `medium`, `default`, `full`, `hopper` (FA3, FlashMLA, FP8 paths), `ep2` (tensor and
expert parallelism on 2 GPUs), `replay` (version-neutral settings for testing old releases).

Every entry records the generated tokens checked and the mismatches. A failing request is recorded
with its prompt length, its first wrong step and the KV positions that step reads. On vLLM the default
matrix also corrupts KV blocks through the real paged cache to prove detection, and maps the probe's
self-describing KV cache to locate the corrupted block.

Each entry runs in its own process. SGLang kills its whole process tree when a scheduler fails (for
example on a backend the GPU cannot compile), and an engine torn down in-process can keep GPU memory the
next entry needs. An entry that crashes or exceeds `--entry-timeout` (default 1800 s) gets an error
record with the end of its output, and the run continues with the next entry. `--in-process` runs every
entry in one process instead.

## Using probec in an engine's CI
A probe twin is an ordinary Hugging Face checkpoint, so compile it once and cache it:

```bash
probec compile --like Qwen/Qwen3-8B --layers 4 -o /cache/probes/qwen3-8b          # CPU, a few minutes at most
probec run /cache/probes/qwen3-8b --engine vllm --matrix quick --strict --out results.jsonl
```

`--strict` exits with status 2 if any generated token differs from the oracle, so the job fails on the
first wrong token. There is nothing to tune: no tolerances, no reference outputs to regenerate.

## How the exactness works (short version)

- Linear layers use only ±1, 16 and powers of two, with at most two terms per output: no rounding.
- Every hidden vector has the same energy, so RMSNorm always divides by exactly 8.
- Softmax, router and logit decisions are won by huge margins (≥ 30–50 nats).
- The tiny residue that survives is far below half a bf16 ulp, so storing results in bf16 snaps them
  back to the designed values ("rounding basins").

Details, per-program constructions and limits: [`docs/DESIGN.md`](docs/DESIGN.md).

## Known deviations of a probe twin from the real model
- Weights are synthetic (bf16); quantization configs are dropped.
- `tie_word_embeddings` is always false; the recall program pins `rope_theta` to 1e12 (posrecall uses the
  real value); MLA twins use dense MLPs in every layer and drop MTP layers.
- Everything else (architecture class, widths, head counts, vocab, norms, RoPE scaling) is the real
  model's, so the engine picks the same kernels.

## Repository layout
- `src/probec`: compiler, programs, oracle, certify, CPU checks, engine harness, CLI.
- `docs/`: design, results, real bugs considered for replay.
- `tools/`: the scripts behind the GPU validation (vast.ai runner, session jobs, the SGLang NGRAM repro).
- `PROGRESS.md`, `BUDGET.md`: the validation log and its GPU spend.

## Credits
The idea of compiling programs into transformer weights comes from Tracr (Lindner et al., 2023,
DeepMind). probec does not use Tracr or RASP; it hand-builds exactness-constrained weights for
production architectures.

## License
Apache-2.0.
