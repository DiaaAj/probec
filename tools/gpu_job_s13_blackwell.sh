#!/usr/bin/env bash
# B200 session 13 (SM100): first Blackwell coverage on vLLM 0.31 (CUTLASS-MLA, FlashInfer-MLA, FA/FlashInfer
# dense, FP8 KV), and the vLLM #52796 replay (base-2 LSE merged as natural log after a prefix hit, MLA prefill
# on SM100; 0.28.0 buggy -> 0.29.0 fixed) with split-marker prompts: the calibrated scale-check head gets
# one marker from the cached context and one from the new tokens, so a wrong merge moves its 4:1 ratio.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
$PY -c "import vllm, torch; print('VERSIONS vllm', vllm.__version__, torch.__version__)" | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.json --layers 3 --program mla -o $P/mla-v3 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json -o $P/recall-sc 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos2 2>&1 | tail -1 | tee -a out/job.log
run(){   # python probe matrix only requests outfile [max_len] [tag]
  log "${8:-v0.31} $(basename $2) $4"
  timeout 1500 $1 -m probec.cli run $2 --engine vllm --matrix $3 --only "$4" --requests $5 --max-len ${7:-8000} --out out/$6 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$4\", \"probe\": \"$(basename $2)\", \"error\": \"exit/timeout\"}" >> out/$6
  python3 -c "import json; r=json.loads(open('out/$6').readlines()[-1]); print(' -> ${8:-v0.31}', r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), r.get('diagnoses'), r.get('spec_decode', ''), [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:6], r.get('error','')[:240])" | tee -a out/job.log
}
for cfg in FLASH_ATTN/default FLASHINFER/default FLASHINFER/fp8kv FLASH_ATTN/ngram_spec_decoys; do run "$PY" $P/recall-sc blackwell $cfg 128 vllm.jsonl 12000; done
run "$PY" $P/pos2 blackwell FLASHINFER/default 128 vllm.jsonl 12000
for cfg in CUTLASS_MLA/default FLASHINFER_MLA/default TRITON_MLA/default CUTLASS_MLA/fp8kv; do run "$PY" $P/mla-v3 blackwell $cfg 64 vllm.jsonl; done
for cfg in default/split_markers default/prefix; do run "$PY" $P/mla-v3 replay $cfg 64 vllm.jsonl; done
for ver in 0.28.0 0.29.0; do
  V=/root/v${ver//./}
  log "install vllm==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  $V/bin/python -c "import vllm, torch; print('VERSIONS', vllm.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  for cfg in default/split_markers default/prefix; do run $V/bin/python $P/mla-v3 replay $cfg 64 replay52796.jsonl 8000 v$ver; done
  rm -rf $V
done
$PY -m probec.cli report out/vllm.jsonl | tail -20 | tee -a out/job.log
log "done"
