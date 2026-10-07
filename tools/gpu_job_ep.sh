#!/usr/bin/env bash
# GPU session (2 GPUs): tensor/expert parallel serving of probe twins, plus the SGLang #32962 replay
# (EPLB with redundant experts summed several times; 0.5.16 buggy -> 0.5.17 fixed).
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/Qwen_Qwen3-30B-A3B.json --layers 4 --program moe -o $P/moe-L4 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json -o $P/recall-sc 2>&1 | tail -1 | tee -a out/job.log

run(){   # engine python probe matrix only requests outfile
  log "$1 $(basename $3) $5"
  timeout 1500 $2 -m probec.cli run $3 --engine $1 --matrix $4 --only "$5" --requests $6 --max-len 8000 --out out/$7 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$5\", \"engine\": \"$1\", \"probe\": \"$(basename $3)\", \"error\": \"exit/timeout\"}" >> out/$7
  python3 -c "import json; r=json.loads(open('out/$7').readlines()[-1]); print(' ->', r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'diag', r.get('diagnoses'), r.get('error','')[:200])" | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 3
}
run vllm "$PY" $P/moe-L4 ep2 FLASH_ATTN/ep2_tp2 96 vllm.jsonl
run vllm "$PY" $P/moe-L4 ep2 FLASH_ATTN/tp2 96 vllm.jsonl
run vllm "$PY" $P/moe-L4 ep2 FLASH_ATTN/ep2_eplb 96 vllm.jsonl
run vllm "$PY" $P/recall-sc ep2 FLASH_ATTN/tp2 96 vllm.jsonl

for ver in 0.5.21 0.5.16 0.5.17; do
  V=/root/sgl${ver//./}
  log "install sglang==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "sglang[all]==$ver" >> out/install_$ver.log 2>&1 || $V/bin/pip install -q "sglang==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  $V/bin/python -c "import sglang, torch; print('VERSIONS sglang', sglang.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  run sglang $V/bin/python $P/moe-L4 ep2 default/ep2_eplb_redundant16 96 sglang_$ver.jsonl
  if [ "$ver" = "0.5.21" ]; then
    run sglang $V/bin/python $P/moe-L4 ep2 default/ep2_tp2 96 sglang_$ver.jsonl
    run sglang $V/bin/python $P/moe-L4 ep2 default/tp2 96 sglang_$ver.jsonl
    run sglang $V/bin/python $P/recall-sc ep2 default/tp2 96 sglang_$ver.jsonl
  fi
  rm -rf $V
done
log "done"
