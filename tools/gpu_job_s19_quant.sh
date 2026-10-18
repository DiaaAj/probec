#!/usr/bin/env bash
# GPU session 19 (RTX 3090): first quantized-path session (M6a, part 1) and the GPU check of the harness fix.
#   1. certify the full Qwen3-0.6B twin (no scale check) in every weight format
#   2. the reference model (HF transformers, on the GPU) generating with fake-quantized activations and
#      FP8 weights, for every activation scheme: does the 28-layer twin stay exact? (docs/QUANTIZED.md §1)
#   3. vLLM 0.31 --matrix quant: online FP8 (FP8 Marlin weight-only on SM86)
#   4. SGLang 0.5.22 --matrix default with every entry in its own process (flex_attention fails on SM86;
#      the FP8 KV and NGRAM entries after it must still run), then --matrix quant
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
log "cpus $(nproc), $(python3 --version 2>&1); $(left) min left before the watchdog"
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
$PY -c "import vllm, torch, transformers; print('VERSIONS vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | tail -1 | logp
$PY -m pip install -q --no-deps -e $EXP/probec 2>&1 | tail -1 | logp
P=$ROOT/probes; mkdir -p $P
$PY -m probec.cli compile --like $EXP/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | logp
[ -f $P/twin/probe_meta.json ] || { log "the twin did not compile; stopping"; exit 1; }

log "1. certify --quant all"
$PY -m probec.cli certify $P/twin --quant all > $O/certify_quant.json 2>&1
$PY -c "
import json; r = json.load(open('$O/certify_quant.json'))
q = r['checks']['quantized_weights']['formats']
print('certify ok', r['ok'], {f: '%d/%d' % (v['exact'], v['tensors']) for f, v in q.items()})" 2>&1 | logp

log "2. reference model with fake-quantized activations (GPU)"
for act in fp8-tensor fp8-token fp8-group128 nvfp4 mxfp4; do
  [ $(left) -lt 60 ] && { log "skip the rest of step 2: $(left) min left"; break; }
  (cd $EXP/probec && timeout 900 $PY -c "
import json
from probec.quant import check_quant
print(json.dumps(check_quant('$P/twin', act='$act', weight='fp8', n_prompts=24, max_len=4000, n_new=64, device='cuda')))
" 2>&1) | grep -v -i -E "warn|it/s" | tail -1 | logp
done

log "3. vLLM --matrix quant"
if [ $(left) -ge 45 ]; then
  (cd $EXP/probec && timeout 1800 $PY -m probec.cli run $P/twin --engine vllm --matrix quant --out $O/vllm_quant.jsonl 2>&1) \
    | grep -v "it/s" > $O/vllm_quant.log
  $PY -m probec.cli report $O/vllm_quant.jsonl 2>&1 | logp
  # leftover vLLM processes (the engine core) would hold GPU memory; match only those
  pkill -9 -f "VLLM::" > /dev/null 2>&1; pkill -9 -f "^$PY -c from multiprocessing" > /dev/null 2>&1; sleep 5
fi

log "4. SGLang 0.5.22"
V=$ROOT/sgl0522
if [ $(left) -ge 40 ]; then
  t0=$(date +%s)
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "sglang[all]==0.5.22" >> $O/install_0.5.22.log 2>&1 || log "install failed (see install_0.5.22.log)"
  log "  install took $(( $(date +%s) - t0 )) s"
  $V/bin/python -c "import sglang; print('VERSIONS sglang', sglang.__version__)" 2>&1 | tail -1 | logp
  for m in default quant; do
    [ $(left) -lt 20 ] && { log "skip SGLang --matrix $m: $(left) min left"; continue; }
    log "  --matrix $m (each entry in its own process)"
    (cd $EXP/probec && PYTHONPATH=src timeout $(( ($(left) - 12) * 60 )) $V/bin/python -m probec.cli run $P/twin --engine sglang \
       --matrix $m --entry-timeout 900 --out $O/sgl_$m.jsonl 2>&1) | grep -v "it/s" > $O/sgl_$m.log
    (cd $EXP/probec && PYTHONPATH=src $V/bin/python -m probec.cli report $O/sgl_$m.jsonl 2>&1) | logp
    pkill -9 -f "^$V/bin/python" > /dev/null 2>&1; sleep 5
  done
fi
log "done ($(left) min left)"
