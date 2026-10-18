# Quantized weights in probec — design for M6 (draft for review)

v1 checks unquantized (bf16) weights. Engines spend much of their kernel code on quantized formats,
whose failure modes are of the kind probec catches: a scale read from the wrong block or group, a wrong
zero point, nibbles unpacked in the wrong order, a per-token scale broadcast along the wrong axis, an
expert's scales swapped with another's. This document proposes how probe twins can stay exact under
quantization, based on measurements of the current twins, and a plan for building it.

## 1. What the current twins already give (measured 2026-10-18 with prototype scripts; §4 step 1 turns them into a command)
**Weights.** In the Qwen3-0.6B recall twin, every linear weight is exactly representable in every
format we checked. That covers all 28 layers' q/k/v/o, gate/up/down, the LM head and the embedding:

| format | grid and scale | groups exact |
|---|---|---|
| FP8 E4M3, amax-based scale per tensor | 448 / amax | all tensors |
| FP8 E4M3, 128×128 blocks (DeepSeek style) | per block | all blocks |
| INT4, group 128, scale and zero point (AWQ) | per group | all groups |
| INT4, group 128, symmetric (GPTQ) | per group | all groups |
| NVFP4: E2M1, E4M3 scale per 16 | per group | all groups |
| MXFP4: E2M1, power-of-two scale per 32 | per group | all groups |

The reason: weights are sparse (at most 0.012 % of a layer weight's entries are nonzero, 0.08 % of the
LM head's) and each tensor uses one or two power-of-two magnitudes (for example up_proj {0.5, 8},
down_proj {0.03125}); no group mixes magnitudes further apart than its format's range. **Weight-only
formats therefore need no change to the probe programs, only checkpoint writers.**

**Activations.** Inputs to the linears of a 2-layer twin of the same shape (3 prompts, 3,398 tokens;
CPU):
- q/k/v inputs (RMSNorm outputs) in both layers, and gate/up inputs in layer 0: exactly {0, 1/8, 2};
  every row's max is 2 (the ballast block), so per-token, per-tensor and per-128-group FP8 scales are
  all the same and every value lands on the E4M3 grid;
- down_proj inputs in layer 0: {0, 32} (saturated SiLU times up);
- attention outputs (o_proj inputs): {0, 1, 2}, plus 1–2 % softmax residue (≈10⁻³⁹, which any
  quantizer flushes to zero);
- **off-grid values** (1.0078, 0.124, 31.75, …) in a few percent of entries. In layer 0 they sit only in
  dims no weight reads (the output of the unused head 1). In layer 1, the last, they also sit in dims the
  weights read (o_proj, up_proj and down_proj inputs). bf16 runs are exact, so those must be at
  positions whose outputs nothing uses, most likely filler tokens, whose lookups are not designed to
  resolve. With amax-based scales an off-grid value only matters if it sets the scale of a meaningful
  value. Per tensor it did not here: the designed maximum is always present and no off-grid value
  exceeded it. Per token or per group it changes only its own token. The twin has only two layers, so
  this argument needs an end-to-end check (§4, step 1) before we rely on it.

## 2. Exactness under quantization
- **Weights:** probec writes the quantized checkpoint itself and picks every scale, so it uses
  power-of-two scales (FP8, MXFP4), E4M3-representable ones (NVFP4) or gcd scales with zero points
  (INT4). Dequantized weights then equal the designed weights bit for bit.
- **Activations:** the engine computes scales from amax at run time (per tensor, per token, per 1×128
  group, per 16 or per 32). The scale need not be a power of two: when every scaled value lands on the
  grid, the only error is the fp32 scale's rounding (≈2⁻²⁴ relative), far inside the bf16 rounding
  basins. Three conditions keep values on the grid:
  1. *pinned maxima:* every quantization group contains its designed maximum (the ballast does this for
     RMSNorm outputs today; attention outputs and down_proj inputs would get a constant channel: a V
     dim that is 2 for every token, a saturated neuron that outputs 32);
  2. *homogeneous groups for FP4:* E2M1's grid {0, ½, 1, 1½, 2, 3, 4, 6} spans a factor of 12, and 1/8 and
     2 differ by 16, so ballast dims and code dims must sit in different 16- and 32-wide groups (the
     compiler aligns the blocks);
  3. *no meaningful off-grid values* (§1).
- **Accumulation:** each designed output sums at most two nonzero products, FP8×FP8 and E2M1×E2M1
  products are exact in fp32, and the products have few significant bits (for example 28 × 448), so the
  sums stay exact even with Hopper's reduced-precision FP8 accumulators.

## 3. Formats, engines and what probec writes
`probec quantize <twin> --format <fmt> -o <dir>` writes the checkpoint layouts vLLM 0.31 and SGLang 0.5.22
load. They were read from both engines' loaders and kernels (2026-10-18), and the module docstring of
`src/probec/quantize.py` has the tensor layouts. Kernels by GPU:

| format (`--format`) | RTX 3090 (SM86) | H100 (SM90) | B200 (SM100) |
|---|---|---|---|
| FP8 per tensor (`fp8`), online FP8 | Marlin W8A16 (both) | CUTLASS W8A8, dynamic per-token activations | CUTLASS (both) |
| FP8 128×128 blocks (`fp8-block`) | Marlin W8A16 (both) | vLLM: FlashInfer + DeepGEMM, scales requantized to powers of two; SGLang: DeepGEMM | DeepGEMM |
| FP8 MoE | vLLM Marlin; SGLang Triton (may not run) | Triton fused MoE | vLLM TRT-LLM / Triton; SGLang Triton |
| AWQ / GPTQ (`int4-awq`, `int4-gptq`) | Marlin | vLLM Machete; SGLang Marlin | Marlin |
| AWQ / GPTQ MoE | Marlin MoE | Marlin MoE | Marlin MoE |
| NVFP4 (`nvfp4`, ModelOpt) | Marlin W4A16 | Marlin W4A16 | W4A4: FlashInfer CuTe-DSL / CUTLASS; TRT-LLM MoE |
| MXFP4 | only gpt-oss MoE (no dense path in either engine) | | |

What the writer has to get right, and what an engine bug in these places would look like:
- **Shared scales across fused shards.** Engines fuse q/k/v and gate/up. vLLM requantizes per-tensor FP8
  shards to their largest scale through fp16, and both engines take the largest NVFP4 global scale
  without rescaling the block scales, which decodes the other shards wrongly. probec gives fused shards
  one power-of-two FP8 scale and one NVFP4 global scale, and gives experts one activation scale per
  layer.
- **Routers stay bf16.** vLLM quantizes a Qwen3-MoE router unless an FP8 or ModelOpt ignore list names
  it exactly; SGLang never quantizes it. probec lists `lm_head` and every `model.layers.N.mlp.gate`.
- **Missing scales load silently.** vLLM's missing-weight check is off for quantized checkpoints: FP8 MoE
  scales default to 1, and NVFP4 per-tensor scales are left uninitialised. A probe catches either
  immediately. A real checkpoint would only drift.
- **Scale direction.** Every stored scale is a decode multiplier, `weight_scale_inv` included.
- **NVFP4's static `input_scale`** comes from a calibration pass over the probe's prompts (amax / (6·448)),
  as ModelOpt's would.

Kernel names are as of vLLM 0.31 and SGLang 0.5.22; each session confirms which kernel actually ran.
The oracle does not change: dequantized weights equal the designed ones, and the designed activations
quantize losslessly, so the expected token stream is the bf16 one. KV-cache quantization (FP8) is already
covered by v1.

## 4. Plan
Status (2026-10-18): step 1 is in `probec certify --quant` (weights) and `probec check --quant-act /
--quant-weight` (`src/probec/quant.py`). On a 2-layer twin without the scale check, every weight is exact in
all six formats and generation stays exact under all five activation schemes (FP8 per tensor, per token
and per 1×128 group, NVFP4, MXFP4). With the scale check on, its calibrated query and its flag MLP's
thresholds need more than FP8's 3 mantissa bits, so quantized twins are compiled with `--no-scale-check`.
The online-FP8 matrices (`--matrix quant`) exist for both engines.

1. **CPU certification (no GPU).** `probec certify --quant <format>`: check that every weight group is
   exact (the §1 table, per twin and program, MoE and MLA included), then run the CPU reference with
   every linear's input and weight fake-quantized in that format and confirm the oracle's tokens. This
   settles the off-grid question of §1. It needs about 2 GB of RAM and a few CPU-minutes per twin. It
   runs on a GPU box or in CI, not on a laptop.
2. **M6a, FP8 (one RTX 3090 session and one H100 session, about $3).** Matrix entries for online FP8
   (vLLM, SGLang), checkpoint writer for FP8 per-tensor and 128-block formats, dense and MoE twins.
   Expected first finding: whether the twin is exact unchanged (weight-only on the 3090) and under W8A8.
3. **M6b, INT4 W4A16 (one RTX 3090 session, about $0.50).** AWQ and GPTQ writers; Marlin dense and MoE.
4. **M6c, FP4 (one B200 session, about $5, and one H100 session).** MXFP4 MoE twin; NVFP4 writer,
   compiler alignment for homogeneous groups, pins.
5. Re-catch: look for fixed bugs in quantized paths of both engines (scale indexing, MoE expert scales,
   group size handling) to replay on pre-fix versions, as v1 did for its bugs.

## 5. Decisions (2026-10-18)
- Order: FP8, then INT4, then FP4 (agreed). MoE and MLA twins run in each step, not as a separate one.
- MXFP4 is deferred: neither engine has a dense MXFP4 path, so it needs a gpt-oss-shaped twin
  (attention sinks, alternating sliding windows, a clamped SwiGLU), which is a program of its own.
- Known quantized-path bugs will be searched for and replayed once the formats run exact.
