#!/usr/bin/env bash
# H100 session 11: FP8 paths that quantize the query — the FP8-robust MLA scale check on FlashMLA FP8 KV
# and DeepSeek-V3.2 (FP8 sparse cache), FA3 FP8 for the dense and positional probes, SGLang FlashMLA / FA3
# FP8; n-gram decoys with draft/accept counters.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.json --layers 3 --program mla -o $P/mla-v3 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.2-Exp.json --layers 3 --program mla -o $P/dsa-v32 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos2 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json -o $P/recall-sc 2>&1 | tail -1 | tee -a out/job.log

run(){   # engine python probe matrix only requests outfile [max_len]
  log "$1 $(basename $3) $5"
  timeout 1500 $2 -m probec.cli run $3 --engine $1 --matrix $4 --only "$5" --requests $6 --max-len ${8:-12000} --out out/$7 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$5\", \"engine\": \"$1\", \"probe\": \"$(basename $3)\", \"error\": \"exit/timeout\"}" >> out/$7
  python3 -c "import json; r=json.loads(open('out/$7').readlines()[-1]); print(' ->', r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), r.get('diagnoses'), r.get('spec_decode', ''), [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:6], r.get('error','')[:240])" | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 2
}
for cfg in FLASHMLA/fp8kv FLASHMLA/default FLASHMLA/ngram_spec_decoys; do run vllm "$PY" $P/mla-v3 hopper $cfg 64 vllm.jsonl 8000; done
run vllm "$PY" $P/dsa-v32 replay default/fp8kv 64 vllm.jsonl 8000
run vllm "$PY" $P/dsa-v32 replay default/prefix 64 vllm.jsonl 8000
for cfg in FLASH_ATTN/fp8kv FLASH_ATTN/ngram_spec_decoys; do run vllm "$PY" $P/recall-sc hopper $cfg 128 vllm.jsonl; done
for cfg in FLASH_ATTN/fp8kv FLASHINFER/fp8kv; do run vllm "$PY" $P/pos2 hopper $cfg 128 vllm.jsonl; done

log "SGLang"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
for cfg in flashmla/fp8kv flashmla/default; do run sglang /root/sgl/bin/python $P/mla-v3 hopper $cfg 64 sglang.jsonl 8000; done
run sglang /root/sgl/bin/python $P/recall-sc hopper fa3/fp8kv 128 sglang.jsonl
run sglang /root/sgl/bin/python $P/pos2 hopper fa3/fp8kv 128 sglang.jsonl
run sglang /root/sgl/bin/python $P/recall-sc hopper flashinfer/ngram_spec_decoys 128 sglang.jsonl
$PY -m probec.cli report out/vllm.jsonl out/sglang.jsonl | tail -30 | tee -a out/job.log
log "done"
