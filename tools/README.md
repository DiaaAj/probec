# tools

Scripts behind the technical report ([`docs/REPORT.md`](../docs/REPORT.md)) and the bug reports. They are not
part of the package.

| Script | Purpose |
|---|---|
| `realdiff.py`, `realdiff_analyze.py` | a real model and its twin under the same engine settings; logprob-tolerance check (report §1) |
| `fault_boundary.py`, `boundary_analyze.py` | faults injected into vLLM's paged KV cache and softmax scale; oracle replay of the read steps (§2.3, §3.2) |
| `sweep_analyze.py` | collect one suite's results across engine releases (§3.3) |
| `bug_costs.py` | the bug timeline and the back-of-envelope cost estimate (§4) |
| `make_figures.py`, `figures_report.py`, `figstyle.py` | every figure in `docs/figures/`, from `docs/data/` |
| `repro_sglang_ngram.py`, `sgl_ngram_realmodel.py` | repro of sglang#44997: NGRAM vs no speculation, on a twin and on real Qwen3-0.6B weights |
| `sgl_repeat.py` | repro of sglang#44999: one request, repeated, under Marlin MoE |
| `marlin_determinism.py` | sglang#45000: SGLang's FP8 Marlin GEMM checked in isolation (sizes, repeats, streams) |
