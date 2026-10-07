#!/usr/bin/env bash
# H100 session: Hopper kernels (FA3, FlashInfer SM90, Triton FP8 KV, FlashMLA, FlashMLA sparse / DSA) on
# probe twins, plus the vLLM #36616 replay (DeepSeek-V3.2 sparse top-k padding: 0.23.0 buggy -> 0.24.0 fixed).
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a out/job.log
$PY -c "import vllm, torch; print('VERSIONS vllm', vllm.__version__, torch.__version__)" | tee -a out/job.log
$PY -c "import deep_gemm; print('deep_gemm ok')" 2>&1 | tail -1 | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json -o $P/recall-sc 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.json --layers 3 --program mla -o $P/mla-v3 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.2-Exp.json --layers 3 --program mla -o $P/dsa-v32 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-30B-A3B.json --layers 4 --program moe -o $P/moe 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log

run(){   # engine python probe matrix only requests outfile [max_len]
  log "$1 $(basename $3) $5"
  timeout 1500 $2 -m probec.cli run $3 --engine $1 --matrix $4 --only "$5" --requests $6 --max-len ${8:-12000} --out out/$7 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$5\", \"engine\": \"$1\", \"probe\": \"$(basename $3)\", \"error\": \"exit/timeout\"}" >> out/$7
  python3 -c "import json; r=json.loads(open('out/$7').readlines()[-1]); print(' ->', r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), r.get('diagnoses'), r.get('error','')[:240])" | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 2
}
for cfg in FLASH_ATTN/default FLASH_ATTN/chunk512_sample FLASHINFER/default TRITON_ATTN/default FLASHINFER/fp8kv TRITON_ATTN/fp8kv FLASH_ATTN/ngram_spec; do
  run vllm "$PY" $P/recall-sc hopper $cfg 128 vllm.jsonl
done
for cfg in FLASH_ATTN/default FLASHINFER/default FLASH_ATTN/chunk512_sample; do run vllm "$PY" $P/pos hopper $cfg 128 vllm.jsonl; done
for cfg in FLASH_ATTN/default FLASH_ATTN/chunk512_sample; do run vllm "$PY" $P/moe hopper $cfg 96 vllm.jsonl 8000; done
for cfg in FLASHMLA/default FLASHMLA/chunk512_sample FLASH_ATTN_MLA/default TRITON_MLA/default FLASHMLA/fp8kv; do
  run vllm "$PY" $P/mla-v3 hopper $cfg 64 vllm.jsonl 8000
done
run vllm "$PY" $P/dsa-v32 replay default/prefix 64 vllm.jsonl 8000
run vllm "$PY" $P/dsa-v32 replay default/shared_prefix 33 vllm.jsonl 8000

for ver in 0.23.0 0.24.0; do
  V=/root/v${ver//./}
  log "install vllm==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  $V/bin/python -c "import vllm, torch; print('VERSIONS', vllm.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  run vllm $V/bin/python $P/dsa-v32 replay default/prefix 64 replay_$ver.jsonl 3000
  run vllm $V/bin/python $P/dsa-v32 replay default/prefix 64 replay_long_$ver.jsonl 8000
  rm -rf $V
done
log "SGLang"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
for cfg in fa3/default fa3/chunk512_sample fa3/page16; do run sglang /root/sgl/bin/python $P/recall-sc hopper $cfg 128 sglang.jsonl; done
for cfg in flashmla/default fa3/default; do run sglang /root/sgl/bin/python $P/mla-v3 hopper $cfg 64 sglang.jsonl 8000; done
run sglang /root/sgl/bin/python $P/moe hopper fa3/default 96 sglang.jsonl 8000
log "SGLang NGRAM baseline on Hopper"
timeout 900 /root/sgl/bin/python /root/exp/probec/tools/sgl_ngram_investigate.py $P/twin A_baseline out/sgl_ngram_hopper.jsonl >> out/sgl_ngram.log 2>&1
timeout 900 /root/sgl/bin/python /root/exp/probec/tools/sgl_ngram_investigate.py $P/twin H_nospec out/sgl_ngram_hopper.jsonl >> out/sgl_ngram.log 2>&1
python3 -c "import json; [print('ngram', r['variant'], r.get('failed_requests'), r.get('bad_tokens')) for r in map(json.loads, open('out/sgl_ngram_hopper.jsonl'))]" | tee -a out/job.log
$PY -m probec.cli report out/*.jsonl | tail -40 | tee -a out/job.log
log "done"
