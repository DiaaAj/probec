# Hybrid linear-attention models in probec — design for M7

Hybrid models interleave linear-attention layers, which keep a fixed-size recurrent state per request,
with ordinary attention layers. Qwen3.5 (0.8B to 397B, dense and MoE) and Qwen3-Next use Gated DeltaNet
(GDN) in three of every four layers; Nemotron-H / Nemotron 3 and Granite 4.0-H use Mamba-2. Engines handle
the recurrent state with code that attention-only models never run:
- a state slot per request, next to the paged KV cache, with its own allocator and its own CUDA-graph
  padding;
- chunked kernels for prefill (FLA's chunked gated delta rule, Mamba-2's SSD) that pass the state from
  chunk to chunk, and a recurrent kernel for decode;
- a short causal convolution in front of the recurrence, with its own rolling state;
- prefix caching that stores states at block boundaries and restores one for a new request;
- speculative decoding that must undo the state updates of rejected draft tokens;
- preemption, which recomputes the state from scratch.

Each of these can corrupt the state silently: a stale or zeroed state, another request's state, a state
from the wrong block boundary, rejected draft tokens left in it. The model still produces fluent text, so
such bugs are hard to catch with accuracy tests. This document describes how a probe twin of a GDN model
keeps every output exact and makes each of these errors change the output.

## 1. The probe: GDN layers as slot memories
The recall program (`docs/DESIGN.md` §2) chases a pointer through the context: in every layer, the
pointer (a ±1 code in the residual) selects the token whose key matches it, and that token's payload
(the code of `nxt(c)` for a class token `c`) becomes the next pointer. The hybrid program keeps this
chain and makes each GDN layer one more hop, using the recurrent state as a table with one slot per class.

**Gated DeltaNet**, per value head, with state `S` (key dim × value dim), for a token with key `k`,
query `q`, value `v` and gates `β`, `g`:

    S ← exp(g) · S;   S ← S + β · k (v − Sᵀk)ᵀ;   o = Sᵀ q · d_k^−½

after a causal depthwise convolution (kernel 4) and SiLU on `q`, `k`, `v`, and an L2 norm on `q` and `k`.
The output `o` goes through a gated RMSNorm (`norm(o) · w · silu(z)`) and `out_proj`.

**Slots.** Qwen3.5 and Qwen3-Next have 16 key heads of dimension 128, so a GDN layer has 2,048 one-hot
keys: exactly as many as the recall program's classes. Each GDN layer assigns every class `c` its own
(key head, key dim) slot through a random permutation (different per layer, so a state read from the
wrong layer finds another class).

**Writes.** Every token carries a static identity field `GK`: code(c) for a class token `c` and for its
echo `e(c)`, and a code that matches no class for fillers and markers. The key projection is a bank of
2,048 detectors, one per slot: detector `j` computes `2·⟨code(c_j), GK⟩ − 88`. That is 40 when the token's
identity is `c_j` and at most −40 otherwise (codes correlate at most 24 of 64). After SiLU, the key is
`40` on the token's own slot and below 10⁻¹⁵ elsewhere, and the L2 norm makes it exactly one-hot. With
`β = σ(32) = 1` and `exp(g) = 1`, the delta rule replaces the slot's row with the new value, exactly:
**every slot holds the value of the latest token written to it.** Fillers and markers write nothing:
their keys are below 10⁻¹³.

**Values.** A class token `c` writes the code of `nxt(c)`. An echo token `e(c)`, the model's output, writes
the code of `nxt²(c)` into the same slot. A value head stores a 64-bit code as 32 bit pairs, each pair as
a one-hot of 4 dims, so exactly 32 of its 128 dims are 40 and the rest 0. The gated RMSNorm then outputs
exactly 2 on those 32 dims (their RMS is a quarter of the head's), `w = 1/64` and `silu(z) = 32` make it 1,
and `out_proj` writes the decoded ±1 code into the register, as an attention hop does.

**Reads.** The query projection is the same bank of detectors on the pointer field, so the query is
one-hot on the pointer's slot, and the head of that slot returns the slot's value. The other key heads
return values below 10⁻⁷, which the residual's rounding absorbs.

**Semantics.** A GDN hop from class `y` goes to `nxt(y)` if the latest token written to `y`'s slot is the
class token `y`, and to `nxt²(y)` if it is an echo `e(y)`. Prompts contain every class of one pool exactly
once (as for recall), so every hop finds a written slot. Attention hops are unchanged (`nxt(y)`; echo
tokens have unique attention keys). The oracle replays the writes in order, and its cost stays linear in
the number of tokens.

**Echo tokens** start their chain at their class: the recall program spends its first layer on reading
the echo token's own key, but in a hybrid model the first layer is a GDN layer. The echo's write and read
in the same step test that the decode kernel writes before it reads.

## 2. What it catches
| state error | what the probe sees |
|---|---|
| lost, zeroed or stale state (a prefix-cache restore at the wrong block boundary, a chunk boundary that drops the state) | classes written in the missing span have empty slots: the hop finds no value and the chain breaks. The decode walk visits every class of the pool, so a missing class is read |
| another request's state | other classes, and other echoes written during that request's decode |
| state of the wrong layer | another permutation: the slot holds a different class's value |
| decode-time writes lost (state not persisted between steps, CUDA-graph padding writing into a real slot) | a class whose echo was emitted is read without its skip, or an unrelated slot is overwritten |
| **rejected draft tokens not rolled back** (n-gram or MTP speculative decoding) | the rejected echo `e(z)` leaves a skip in `z`'s slot; the next hop through `z` goes to `nxt²(z)` instead of `nxt(z)` |
| preemption that recomputes the state wrongly | any of the above |

The KV-cache errors that recall catches are still caught in the attention layers.

**Not covered in v1:**
- *the causal convolution's state:* the probe uses only the current token's tap, so a wrong conv state
  changes nothing;
- *decay and partial writes:* `exp(g)` is pinned to 1 and `β` to 1, so a kernel that mishandles other
  values is not exercised;
- *Mamba-2* (§5).

## 3. Exactness
| op | why it is exact |
|---|---|
| in_proj (q, k, v detectors, z, b, a) | each output is a 1- or 2-term dot product of ±1/8 inputs and power-of-two weights plus a constant from the ballast: small integers, exact in bf16 |
| causal conv1d | weight `[0, 0, 0, 1]`: the output is the current input; SiLU(40) = 40 and SiLU(≤ −40) < 10⁻¹⁵ in fp32 |
| L2 norm of q, k | `40 / √(1600 + 10⁻⁶)` rounds to exactly 1 in fp32; the other entries stay below 10⁻¹³ |
| gates | `β = σ(32)` is 1 in fp32; `g = −exp(A_log) · softplus(−32)` is 0 or −10⁻¹⁴, so `exp(g)` and its chunked cumulative products are exactly 1 |
| delta rule, chunked or recurrent | one-hot keys make `Sᵀk` one row of `S`; values are 0 or 40, differences 0 or ±40; the chunked form's triangular solve has entries in {0, ±1}. Every product is exact in bf16 or fp32 |
| query scale `d_k^−½` and the read | `o = 0.0884 · v` is inexact (≈2⁻²⁴ relative), but the gated RMSNorm divides it out: the normalized value is 2 to within 10⁻⁶, and bf16 rounds it to exactly 2 |
| gated RMSNorm, out_proj, residual | outputs on {0, 1}; the register receives exactly ±1, as from an attention hop |
| attention layers (gated) | recall's attention with the output gate at `σ(32) = 1`; codes sit in the dims RoPE does not rotate (partial rotary factor 0.25), so lookups are position-invariant without changing rope_theta; zero-centered norms get weight 0 |

Leakage (the 10⁻¹³ keys of fillers, the 10⁻⁷ reads of unused heads) stays below a thousandth of half a
bf16 ulp of the register's ±1 values, so the residual stays canonical, and the bit-exact internal-state
check verifies this.

## 4. Engine notes (vLLM 0.31, SGLang 0.5.22; read from the source at those tags)
- **Kernels.** Both engines run FLA-derived Triton kernels: chunked prefill (chunk size 64), a packed recurrent
  kernel for decode, and a sigmoid-gating kernel for mixed and speculative batches; vLLM also has a CUDA
  kernel for speculative verification and FlashInfer prefill on SM90 and SM100. Their L2 norms differ in
  form (`x / √(Σ+ε)`, `x · (1/√(Σ+ε))`, `x · rsqrt(Σ+ε)`). With the detector's 32, Σ = 1024 and every form
  gives exactly 1. In prefill both engines round the normalized q and k to bf16 first, and SGLang rounds
  β to bf16 as well, which is exact for 1. GDN prefill asserts bf16 or fp16 inputs: an fp32 twin cannot run.
- **State.** fp32 for Qwen3.5 in both engines (from the config's `mamba_ssm_dtype`), the conv state in bf16.
  vLLM keeps one state block per speculative position and reads the state of the last accepted token;
  SGLang verifies without updating the state and copies the accepted step's state afterwards.
- **Prefix caching.** vLLM turns it on by default for hybrids. It caches states at block boundaries
  (`mamba_cache_mode` "align", which also forces chunked prefill), with attention blocks enlarged to the
  state's page size. With CUDA graphs on, `max_num_seqs` may not exceed the number of state blocks. SGLang's
  radix cache covers these models too; it keeps extra state buffers when the overlap scheduler is on
  (`--mamba-radix-cache-strategy` extra_buffer) and none without it (no_buffer).
- **Config fields.** SGLang derives layer types from `full_attention_interval` alone and reads
  `partial_rotary_factor` at the top level; vLLM reads `layer_types` and `rope_parameters`. The twin writes
  both and refuses non-periodic layer patterns. vLLM raises on missing or unexpected weights; SGLang skips
  unknown names with a warning, so a naming error there would only show as wrong output.

## 5. Plan
1. **Done (2026-10-18):** shapes and configs: Qwen3.5 dense and MoE text models from real configs (the twin
   drops the vision tower and the MTP head); built-in tiny shapes for CPU tests.
2. **Done:** the `hybrid` program and its oracle, CPU-checked through HF transformers (chunked prefill and
   recurrent decode: 0 mismatches, bit-exact internal states); the state-fault campaign; MoE hybrids with
   the routed experts at zero and the pointer logic in the gated shared expert.
3. **GPU session 25:** vLLM 0.31 and SGLang 0.5.22 on an RTX 3090 with the full-depth Qwen3.5-0.8B twin
   (`--matrix hybrid`: default, attention backends, prefix caching with shared prefixes, chunked prefill,
   preemption, state dtype, decode kernels, n-gram speculation with decoys, FP8 KV), plus 4B, 27B and
   35B-A3B twins at reduced depth.
4. Qwen3-Next (fused, per-key-head interleaved `in_proj_qkvz` / `in_proj_ba`; MoE in every layer).

## 6. Later
- **Mamba-2 hybrids** (Nemotron-H, Granite 4.0-H): the same slots with `B` (keys) and `C` (queries) one-hot
  per group (`n_groups × d_state` slots, 1,024 for Nemotron 3 Nano), `Δ = softplus(32) = 32` and `A ≈ 0`.
  The Mamba-2 state adds writes instead of replacing them, so echo writes cannot override class writes;
  v1 for Mamba-2 would check state membership only.
- **Conv-state coverage:** a "herald" variant in which every class token's value comes through the
  convolution's previous-token tap from a preceding herald token, so a wrong conv state at a chunk or
  prefix boundary loses the write.
- Kimi Linear (KDA + MLA), MiniMax lightning attention.
