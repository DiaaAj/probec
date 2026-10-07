#!/usr/bin/env bash
# GPU session 7 (RTX 3090): (1) isolate the MLA probe failures seen on vLLM 0.25.1/0.26.0 (same workload,
# config variants) and bisect releases; (2) replay #42642 with long suffixes and #47728 with async
# scheduling; (3) localization retest (all copies); (4) positional probe v2 on vLLM 0.31.
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
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V2-Lite.json --layers 6 --program mla -o $P/mla-v2lite 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/mistralai_Mistral-7B-v0.1.json --layers 4 -o $P/swa-mistral 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos2 2>&1 | tail -1 | tee -a out/job.log

run(){   # python probe matrix only requests max_len outfile tag
  log "$8 $(basename $2) $4"
  timeout 1200 $1 -m probec.cli run $2 --engine vllm --matrix $3 --only "$4" --requests $5 --max-len $6 --out out/$7 >> out/run.log 2>&1
  python3 -c "import json; r=json.loads(open('out/$7').readlines()[-1]); print(' -> $8', r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), 'fails', [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:8], r.get('error','')[:200])" | tee -a out/job.log
}
# MLA failure isolation on the image's vLLM 0.31 (replay workload: rng 0, 96 requests, max_len 3700)
run "$PY" $P/mla-v2lite replay default/prefix 96 3700 mla31.jsonl v0.31
for v in mla/default mla/no_prefix_cache mla/eager mla/big_batch_budget mla/small_batch_budget mla/no_chunked_prefill; do
  run "$PY" $P/mla-v2lite mladebug $v 96 3700 mla31.jsonl v0.31
done
log "localization retest"
timeout 1200 $PY -m probec.cli run $P/twin --engine vllm --only FLASH_ATTN/default --out out/loc31.jsonl >> out/loc.log 2>&1
python3 -c "import json; r=json.loads(open('out/loc31.jsonl').readlines()[-1]); print('loc31', json.dumps(r.get('targeted_faults', r.get('targeted_faults_error'))))" | tee -a out/job.log
for cfg in FLASH_ATTN/default FLASHINFER/default TRITON_ATTN/default; do run "$PY" $P/pos2 quick $cfg 96 8000 pos31.jsonl v0.31; done 2>/dev/null
run "$PY" $P/pos2 default FLASH_ATTN/chunk512_sample 96 8000 pos31.jsonl v0.31

for ver in 0.29.0 0.30.0 0.27.1 0.28.0 0.25.1 0.26.0; do
  V=/root/v${ver//./}
  log "install vllm==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  run $V/bin/python $P/mla-v2lite replay default/prefix 96 3700 bisect.jsonl v$ver
  if [ "$ver" = "0.25.1" ] || [ "$ver" = "0.26.0" ]; then
    run $V/bin/python $P/mla-v2lite replay default/shared_prefix_long_suffix 33 3700 replay.jsonl v$ver
    run $V/bin/python $P/swa-mistral replay default/ngram_spec_async 96 12000 replay.jsonl v$ver
  fi
  rm -rf $V
done
log "done"
