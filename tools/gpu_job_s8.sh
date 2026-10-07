#!/usr/bin/env bash
# GPU session 8 (RTX 3090): re-validate the positional probe v2 (identical-token prefix) on vLLM 0.31 and
# SGLang 0.5.21, and check the maintainer-facing SGLang NGRAM repro script end to end.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos2 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log
run(){ log "$1 $3"; timeout 1200 $2 -m probec.cli run $P/pos2 --engine $1 --matrix $4 --only "$3" --requests 128 --out out/$1.jsonl >> out/run.log 2>&1
  python3 -c "import json; r=json.loads(open('out/$1.jsonl').readlines()[-1]); print(' ->', r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'req', r.get('mismatched_requests'), r.get('error','')[:200])" | tee -a out/job.log; pkill -f sglang >/dev/null 2>&1; sleep 2; }
for cfg in FLASH_ATTN/default FLASH_ATTN/chunk512_sample FLASH_ATTN/eager_preempt FLASHINFER/default FLASHINFER/chunk512_sample FLASHINFER/fp8kv TRITON_ATTN/default FLASH_ATTN/ngram_spec; do
  run vllm "$PY" $cfg default
done
log "SGLang"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
for cfg in flashinfer/default triton/default flashinfer/chunk512_sample; do run sglang /root/sgl/bin/python $cfg default; done
log "NGRAM repro script"
cd /root/exp/probec && timeout 1500 /root/sgl/bin/python tools/repro_sglang_ngram.py $P/twin 2>&1 | grep -v "it/s" | tail -12 | tee -a /root/exp/out/job.log
cd /root/exp && log "done"
