# probec v1 plan

**Goal.** An open-source Python tool that compiles exact known-answer "probe twins" of real model
architectures and runs them through unmodified vLLM and SGLang. It catches cases where the engine reads
the wrong data (wrong KV block, slot or layer; tokens sent to the wrong expert; bad MLA reads), with no
false alarms, and says where the fault is.

**Not in v1:** hybrid/SSM models, quantized weights, TRT-LLM, llama.cpp, performance testing.

## Milestones

| | What | Done when |
|---|---|---|
| M0 | Package the prototype: compiler, oracle, CPU checks, CLI, tests | CLI reproduces the prototype on CPU |
| M1 | Dense probe as a tool: twins from real configs (Llama, Qwen3 with QK-norm), positional probe at realistic rope_theta, SGLang adapter, GPU localization, scale-bug experiment | 0 mismatches over >= 1M tokens per engine; targeted corruptions caught and located |
| M2 | MoE probe: expert tagging on a Qwen3-MoE twin; one GPU, then expert parallelism | injected routing faults caught and named; 0 false alarms |
| M3 | MLA probe: DeepSeek-style MLA (sparse variant on Hopper); re-catch real past bugs | pre-fix versions flagged, post-fix clean |
| M4 | Release: validation, docs (credit Tracr), CPU CI, public repo | first public version |

## Working rules
- GPU sessions are capped in time and cost, instances are always destroyed, spend goes in `BUDGET.md`.
- When the GPU budget runs out, CPU work continues and GPU tests are parked with exact run commands.
- Progress is logged in `PROGRESS.md` at every milestone step.
