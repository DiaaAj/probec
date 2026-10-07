# probec design

probec compiles small programs into the weights of a real model architecture so that an inference
engine, running its real kernels, must produce one exact, known token stream. This document explains
how the exactness works and what each probe program checks.

## 1. Why the outputs are exact

Floating-point inference is never exact in general, so probec does not try to make arithmetic exact.
It designs weights so that the answer does not depend on the tiny errors every kernel makes.

| Step | Normally inexact because… | probec's construction |
|---|---|---|
| Matrix multiplies | sums are rounded | every value is ±1, 16 or a power of two, and each output sums at most two terms |
| RMSNorm | divides by a square root | every hidden vector has energy exactly 64·hidden (±1 codes plus a ballast block at 16), so it always divides by exactly 8; the epsilon vanishes below fp32 resolution |
| Attention softmax | exp() | the right key wins by ≥ 30–50 nats, so every other key's weight is ≤ e⁻³⁰ |
| SiLU gates | sigmoid() | gates are saturated: silu(32) = 32 exactly in fp32 |
| RoPE | sin/cos | content codes sit in RoPE pairs that barely rotate (rope_theta 1e12), or in MLA's no-RoPE part |
| Sampling | randomness | the right token wins by ≥ 32 nats, so greedy and T=1 sampling agree |

Whatever residue survives (≈10⁻²¹ after attention, ≈10⁻¹³ after a saturated gate) is far below half a
bf16 ulp. Storing the result in bf16 snaps it back to the designed value: every designed value sits in
the middle of its *rounding basin*. Values that must stay exact are ±1-coded rather than 0/1-coded,
because floating point has no basin around zero.

`probec certify` checks the static conditions: every weight is a small dyadic rational (except tensors
that only shape scores), every embedding row has energy exactly 64·hidden, and the score, logit and
router margins are large enough.

## 2. Programs

### recall (dense GQA: Llama, Mistral, Qwen2, Qwen3)
Pointer chasing through the KV cache. Each class token carries a key code and a payload code; each
layer's attention head 0 looks up the token whose key equals the current pointer and copies its
payload; the MLP moves the payload into the pointer. After L layers the model emits an "echo" token
naming where the chase ended. Echo tokens have unique keys, so every decode step's layer 0 must read
the newly written KV of the current token. Keys carry per-layer ±1 masks, so a wrong-layer read finds
nothing.

**Scale check.** Two marker tokens are inserted into every prompt. In the last layer, a second head
gives both markers scores whose difference is exactly ln 4 at the model's softmax scale, so it returns
0.8 − 0.2 = 0.6. The MLP turns that into FLAG = +1 when 0.5 < value < 0.7 and −1 otherwise, and the
LM head gives a dedicated token `S_err` the logit −100·FLAG. A softmax-scale fault outside about
0.75×–1.35× (a dropped or doubled 1/√d, exp vs exp2, a wrong key scale) therefore makes the model emit
`S_err`. The 4:1 ratio keeps P values exact even for kernels that quantize P to FP8.

### posrecall (RoPE and positions at the real rope_theta)
Chained relative-offset reads with content-free, RoPE-only scores: keys are one constant vector, and
head h's query is that vector pre-rotated by −δ_h·ω_i in every RoPE pair, so q·k peaks at relative
position δ_h. An accumulator A, the elementwise product of every code read so far, picks which head's
read becomes the pointer (log2 H bits of A index the head). Because A remembers the whole read history,
two read chains that diverged never merge again: a single wrong read is masked only if the final read
happens to coincide. The compiler computes the real RoPE frequencies (llama3 scaling included) and sizes
the query amplitude so that the peak beats every other offset in the usable context by ≥ 30 nats. A
position-id slip, a RoPE frequency error or a key cached with the wrong position changes which token is
read.

Prompts start with max-offset copies of one class token. A read whose offset reaches before position 0
can only happen inside that prefix, where the head averages identical values, which is exact. (Without
the prefix, such mixed reads perturb early tokens' residuals, the accumulator compounds the error over
many layers until it exceeds the bf16 rounding basins, and later reads of early tokens go wrong; the
first GPU runs of v2 hit exactly this.)

### moe (Qwen3-MoE)
The recall program with an MoE block in every layer. A ROUTE field in each token's embedding fixes a
k-subset of experts (router logits ±16, so top-k is exact and the renormalized weights are exactly
1/k). Every expert repeats the pointer logic, so the k weighted outputs sum to exactly one update: a
dropped or duplicated expert breaks the pointer. Expert e also flips its own TAG bit; the next layer
compares TAG with −ROUTE and, on a mismatch for expert j, raises DIAG[j] (diagnostic token D_j wins
with logit 100) and corrupts the token's key. Misrouted and duplicated experts are named by the
diagnostic token; the last layer's routing is unchecked.

### mla (DeepSeek V2/V3 multi-head latent attention)
The recall program on MLA. The cached latent holds the masked key code, the payload code and +1 fill,
so every latent entry is ±1, its RMS is exactly 1 and bf16/FP8 caches hold it exactly. kv_b_proj
expands head 0's no-RoPE key and value; RoPE parts are zero, so lookups are position-invariant at the
model's real rope settings (YaRN included). Engines that absorb W_UK/W_UV see the same exact values.
The twin uses dense MLPs in every layer (first_k_dense_replace = num_layers).

**DSA (DeepSeek V3.2).** The indexer's head 0 scores the masked pointer against the masked key code, so
the token each layer reads is the indexer's top-1. Its key puts the code in the no-RoPE half and the
negated code in the RoPE half: mean 0 and variance 1, so the indexer's LayerNorm is the identity.

**Scale check.** As in recall, on head 1 of the last layer. Two ±1 latent dims carry "is a marker"
(MA + MB + 1) and "A or B" (−MB). kv_b_proj forms k₀ = A-or-B, kᵢ = 2·is-marker + 2 for i ≥ 1 (4 on the
markers, 0 elsewhere) and v = A-or-B; the query (c, 2, …, 2) comes from a +1 fill dim of the q latent,
with c = ln 4 / (2·softmax scale). The A-versus-B difference sits alone in its latent dim, so the
absorbed query (W_UKᵀq, which FP8 MLA kernels quantize to FP8 along with the cache) still carries c
to within FP8's relative precision. An earlier version mixed c into the large marker term; FP8
quantization erased it and the check fired on every request with FlashMLA's FP8 KV cache.
Under DSA the calibrated head only sees the indexer's top-k set, so two more indexer heads
query ±code(marker A) and marker B's code is −code(A). Each marker scores 64 on its own head, while any
other key gains ReLU(x) + ReLU(−x) = |x| ≤ 24 from the pair (the compiler searches for a marker code
with that bound): the read token and both markers always make the top-k set.

## 3. Localization
The oracle knows which prompt positions every generation step reads. When the first wrong token
appears at step s, the KV blocks holding those positions are the suspects. On vLLM, the probe's KV
cache is self-describing: the sign pattern of every cached key in layer 0 names the token stored in
that slot, so probec maps physical blocks to request blocks without engine internals and can corrupt
a chosen block to prove detection and localization.

## 4. Workloads
- **default:** random pools and lengths (70% short, 25% up to 6,000 tokens, 5% longer); 30% of requests
  share a long prefix with an earlier one, to exercise prefix caching.
- **shared / shared_long:** one long base prompt and variants sharing it with short (1–64) or long
  (600–1,200) suffixes, run after the base warms the cache: cached-context prefill paths.
- **decoys:** probe outputs rarely repeat, so n-gram drafters seldom propose anything. Decoy prompts carry
  fragments of their own expected output over filler runs; half continue correctly (accepted drafts) and
  half diverge (rejected drafts). Echo tokens are never read, so the expected output is unchanged.
  Shared-prefix variants inherit their base's decoys, so they still hit the prefix cache.
- **split_markers:** a base prompt warms the cache; each variant replaces the later scale-check marker
  with a filler and re-inserts it into a long new suffix. Engines that prefill the suffix against the
  cached context compute two partial attentions and merge them by log-sum-exp; the calibrated head then
  gets one marker from each side, so a mis-scaled merge (e.g. base-2 LSE taken as natural log: ×1.44 or
  ×0.69 on the score gap) moves its 4:1 ratio out of the window. The main lookups cannot see such a
  fault: their winner sits in one partial with a 30+ nat margin.

## 5. What probec cannot see
- numeric bugs that keep large-margin outcomes (accumulation order, slightly wrong scales) — the
  rounding basins hide them by design (the scale check covers gross softmax-scale faults);
- quantized-weight numerics (FP8/NVFP4/INT4 weights are not part of v1);
- value-dependent kernel branches that probe values never trigger;
- text-level bugs (tokenizer, chat template, parsers).
