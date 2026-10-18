#!/usr/bin/env bash
# GPU session 21 (RTX 3090): quantized checkpoints on Ampere (M6a/M6b), where every format runs weight-only
# through Marlin: FP8 per-tensor and 128x128 blocks (W8A16), AWQ and GPTQ (W4A16), NVFP4 (W4A16), for a dense
# and an MoE twin (Marlin MoE), on vLLM 0.31 and SGLang 0.5.22, every entry in its own process. The MoE AWQ
# checkpoint that session 20 could not write (an INT4 zero-point bug, fixed) and MoE NVFP4 (per-expert
# calibration) are new here.
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
compile(){ $PY -m probec.cli compile --like $D/$1 --no-scale-check "${@:3}" -o $P/$2 2>&1 | tail -1 | sed "s/^/$2: /" | logp; }
compile Qwen_Qwen3-0.6B.json dense
compile Qwen_Qwen3-30B-A3B.json moe --layers 2 --program moe
for t in dense moe; do
  $PY -m probec.cli certify $P/$t --quant all > $O/certify_$t.json 2>&1
  $PY -c "
import json; r = json.load(open('$O/certify_$t.json')); q = r['checks']['quantized_weights']['formats']
print('$t certify', {f: '%d/%d' % (v['exact'], v['tensors']) for f, v in q.items()})" 2>&1 | logp
done
for spec in dense:fp8 dense:fp8-block dense:int4-awq dense:int4-gptq dense:nvfp4 moe:fp8 moe:fp8-block moe:int4-awq \
            moe:int4-gptq moe:nvfp4; do
  t=${spec%%:*}; f=${spec#*:}
  $PY -m probec.cli quantize $P/$t --format $f --device cuda -o $P/$t-$f 2>&1 | grep -v -i warn | tail -1 | logp
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
for c in dense-fp8 dense-fp8-block dense-int4-awq dense-int4-gptq dense-nvfp4 moe-fp8 moe-fp8-block moe-int4-awq \
         moe-int4-gptq moe-nvfp4; do
  run v_$c $PY vllm $P/$c quick FLASH_ATTN/default
done

log "SGLang 0.5.22"
V=$ROOT/sgl0522
if [ $(left) -ge 25 ]; then
  t0=$(date +%s)
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "sglang[all]==0.5.22" >> $O/install_0.5.22.log 2>&1 || log "install failed (see install_0.5.22.log)"
  log "  install took $(( $(date +%s) - t0 )) s"
  for c in dense-fp8 dense-fp8-block dense-int4-awq dense-int4-gptq dense-nvfp4 moe-fp8 moe-fp8-block moe-int4-awq \
           moe-int4-gptq moe-nvfp4; do
    run s_$c $V/bin/python sglang $P/$c quick flashinfer/default
  done
fi
log "done ($(left) min left)"
