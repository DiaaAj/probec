#!/usr/bin/env bash
# GPU session 25c (RTX 3090): the hybrid twins on SGLang 0.5.22 (session 25 lost its results; it never reached
# SGLang). The full hybrid matrix on the Qwen3.5-0.8B twin and the quick entry on the 4B, 27B, 35B-A3B and
# Qwen3-Next twins.
# Environment: CREATED = the instance's creation time (epoch seconds) as the launcher saw it.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=/root; EXP=/root/exp; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }
CAP_MIN=140
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
compile Qwen_Qwen3.5-4B.json h4b --layers 8
compile Qwen_Qwen3.5-27B.json h27b --layers 4
compile Qwen_Qwen3.5-35B-A3B.json h35moe --layers 4
compile Qwen_Qwen3-Coder-Next.json hnext --layers 4
for t in h08 h4b h27b h35moe hnext; do
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
log "SGLang 0.5.22"
V=$ROOT/sgl0522
t0=$(date +%s)
python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
$V/bin/pip install -q -U pip >/dev/null 2>&1
$V/bin/pip install -q "sglang[all]==0.5.22" >> $O/install_0.5.22.log 2>&1 || log "install failed (see install_0.5.22.log)"
$V/bin/pip install -q --no-deps -e $EXP/probec >> $O/install_0.5.22.log 2>&1
log "SGLang 0.5.22 installed in $(( $(date +%s) - t0 )) s"
run s_h08 $V/bin/python sglang $P/h08 hybrid
for t in h4b h27b h35moe hnext; do run s_$t $V/bin/python sglang $P/$t quick; done
log "done ($(left) min left)"
