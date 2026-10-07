#!/usr/bin/env bash
# GPU session 12 (RTX 3090): is vLLM's multi-chunk MLA context path still wrong? #50613 fixed the failure
# by fitting a batch's cached context into the prefill workspace in one chunk when possible; these
# workloads put more cached context in a step than the workspace holds. vLLM 0.31 vs 0.27.1 (control).
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
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V2-Lite.json --layers 6 --program mla --no-scale-check -o $P/mla-v2lite 2>&1 | tail -1 | tee -a out/job.log
run(){   # python matrix only requests max_len outfile tag
  log "$7 $3"
  timeout 1500 $1 -m probec.cli run $P/mla-v2lite --engine vllm --matrix $2 --only "$3" --requests $4 --max-len $5 --out out/$6 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$3\", \"error\": \"exit/timeout\"}" >> out/$6
  python3 -c "import json; r=json.loads(open('out/$6').readlines()[-1]); print(' -> $7', r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:8], r.get('error','')[:240])" | tee -a out/job.log
}
suite(){   # python tag
  run $1 mladebug mla/shared_long_big_budget 65 12000 chunks.jsonl $2
  run $1 mladebug mla/big_budget_many_seqs 128 12000 chunks.jsonl $2
  run $1 replay default/shared_prefix_long_suffix 65 12000 chunks.jsonl $2
  run $1 replay default/prefix 128 12000 chunks.jsonl $2
}
suite "$PY" v0.31
log "install vllm==0.27.1"
python3 -m venv /root/v27 >/dev/null 2>&1 || $PY -m venv /root/v27
/root/v27/bin/pip install -q -U pip >/dev/null 2>&1
/root/v27/bin/pip install -q "vllm==0.27.1" >> out/install_0.27.1.log 2>&1
/root/v27/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_0.27.1.log 2>&1
suite /root/v27/bin/python v0.27.1
log "done"
