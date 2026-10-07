# tools

Scripts used to validate probec on rented GPUs (vast.ai) and to investigate findings. They are not
part of the package.

| Script | Purpose |
|---|---|
| `vast_run.sh` | rent an offer, push a run directory, run a command, pull `out/`, always destroy (hard time cap) |
| `gpu_job_*.sh` | the GPU sessions behind `PROGRESS.md` (engine matrices, Hopper, expert parallel, release replays) |
| `gpu_job_s9b.sh` | example of bisecting a vLLM failure to one commit with per-commit wheels (`wheels.vllm.ai/<sha>`) |
| `sgl_ngram_investigate.py` | rerun the SGLang NGRAM workload under one configuration variant, record failing requests |
| `analyze_ngram.py` | explain failing requests against the oracle |
| `repro_sglang_ngram.py` | minimal repro: SGLang NGRAM vs no speculation (and vs the oracle) on a probe |
| `sgl_ngram_realmodel.py` | the same comparison on real Qwen3-0.6B weights |
