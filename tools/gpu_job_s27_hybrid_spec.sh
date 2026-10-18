#!/usr/bin/env bash
# GPU session 27 (RTX 3090): isolate vLLM 0.31's n-gram speculative-decoding failures on the hybrid Qwen3.5-0.8B
# twin (session 26a: 8-20 of 192 requests wrong, every one explained by an accepted token's GDN write missing
# from the state; SGLang exact on the same probe). Same prompts (seed 12) with one setting changed per entry:
# repeat, async scheduling off, 1 and 2 speculative tokens, eager, one sequence at a time, the GPU n-gram
# drafter, and no decoys.
# Environment: CREATED = the instance's creation time (epoch seconds) as the launcher saw it.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=/root; EXP=/root/exp; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }
CAP_MIN=110
NOW=$(date +%s)
if [ -n "${CREATED:-}" ] && [ "$CREATED" -le $((NOW + 300)) ] 2>/dev/null && [ $((NOW - CREATED)) -lt $((CAP_MIN * 60)) ]; then
  DEADLINE=$((CREATED + CAP_MIN * 60))
else
  DEADLINE=$((NOW + CAP_MIN * 60 - 30 * 60))
fi
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1 | logp
log "cpus $(nproc); $(left) min left before the watchdog"
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
$PY -c "import vllm, torch, transformers; print('VERSIONS vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | tail -1 | logp
$PY -m pip install -q --no-deps -e $EXP/probec 2>&1 | tail -1 | logp
D=$EXP/probec/tests/data; P=$ROOT/probes; mkdir -p $P
compile(){ $PY -m probec.cli compile --like $D/$1 "${@:3}" -o $P/$2 2>&1 | tail -1 | sed "s/^/$2: /" | logp; }
compile Qwen_Qwen3.5-0.8B.json h08
for t in h08; do
  $PY -m probec.cli certify $P/$t > $O/certify_$t.json 2>&1
  $PY -c "import json; r = json.load(open('$O/certify_$t.json')); print('$t certify ok', r['ok'], {k: v['ok'] for k, v in r['checks'].items()})" 2>&1 | logp
done
run(){   # tag python engine probe matrix [only]
  local tag=$1 py=$2 eng=$3 probe=$4 m=$5 only=${6:-}
  [ $(left) -lt 12 ] && { log "skip $tag: $(left) min left"; return; }
  log "$tag: $eng --matrix $m ${only:+--only $only} on $(basename $probe)"
  (cd $EXP/probec && PYTHONPATH=src timeout $(( ($(left) - 8) * 60 )) $py -m probec.cli run $probe --engine $eng --matrix $m \
     ${only:+--only $only} --entry-timeout 900 --out $O/$tag.jsonl 2>&1) | grep -v "it/s" > $O/$tag.log
  (cd $EXP/probec && PYTHONPATH=src $py -m probec.cli report $O/$tag.jsonl 2>&1) | grep -E "^\| (vllm|sglang|\?)|configurations" | logp
}
log "vLLM 0.31"
run v_h08_spec $PY vllm $P/h08 hybridspec
log "done ($(left) min left)"
