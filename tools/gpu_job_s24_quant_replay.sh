#!/usr/bin/env bash
# GPU session 24 (RTX 3090): replay two fixed vLLM quantized-path bugs on their last buggy release.
#   vLLM #44735 (0.23.0 -> 0.24.0): FP8 Marlin (SM < 89) skips the weight transpose on square layers. The
#     Qwen3-8B-shaped twin's o_proj is 4096 x 4096; the Qwen3-0.6B shape has no square layer (control).
#     Probe-written FP8 checkpoint and online quantization="fp8".
#   vLLM #34577 (0.18.1 -> 0.19.0): NVFP4 Marlin's S0E5M3 scale trick underflows for small block scales,
#     the bug SGLang 0.5.22 still has (sessions 20, 21). probec's unit-scale NVFP4, dense and MoE;
#     ModelOpt-style scales as the control.
# Environment: CREATED = the instance's creation time (epoch seconds) as the launcher saw it.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=/root; EXP=/root/exp; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }
CAP_MIN=105
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
$PY -m pip install -q --no-deps -e $EXP/probec 2>&1 | tail -1 | logp
D=$EXP/probec/tests/data; P=$ROOT/probes; mkdir -p $P
compile(){ $PY -m probec.cli compile --like $D/$1 --no-scale-check "${@:3}" -o $P/$2 2>&1 | tail -1 | sed "s/^/$2: /" | logp; }
compile Qwen_Qwen3-0.6B.json dense
compile Qwen_Qwen3-8B.json dense8b --layers 4
compile Qwen_Qwen3-30B-A3B.json moe --layers 2 --program moe
q(){ $PY -m probec.cli quantize $P/$1 --format $2 --device cuda -o $P/$3 "${@:4}" 2>&1 | grep -v -i warn | tail -1 | logp; }
q dense8b fp8 dense8b-fp8
q dense fp8 dense-fp8
q dense nvfp4 dense-nvfp4
q dense nvfp4 dense-nvfp4-mo --nvfp4-scales modelopt
q moe nvfp4 moe-nvfp4

install(){   # version -> venv path on stdout
  local V=$ROOT/vllm$1 t0=$(date +%s)
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$1" >> $O/install_$1.log 2>&1 || log "install vllm==$1 failed (see install_$1.log)"
  $V/bin/pip install -q --no-deps -e $EXP/probec >> $O/install_$1.log 2>&1
  log "vllm $($V/bin/python -c 'import vllm, torch; print(vllm.__version__, "torch", torch.__version__)' 2>&1 | tail -1) installed in $(( $(date +%s) - t0 )) s"
}
run(){   # version tag twin entry
  local V=$ROOT/vllm$1
  [ $(left) -lt 12 ] && { log "skip $1 $2: $(left) min left"; return; }
  (cd $EXP/probec && PYTHONPATH=src timeout 1200 $V/bin/python -m probec.cli run $P/$3 --engine vllm --matrix replay --only $4 \
     --entry-timeout 900 --out $O/$1_$2.jsonl 2>&1) | grep -v "it/s" > $O/$1_$2.log
  grep -h -o "Selected [A-Za-z0-9]*Kernel[A-Za-z0-9]*\|Using [A-Za-z0-9]*Marlin[A-Za-z0-9 ]*" $O/$1_$2.log | sort | uniq -c | head -3 | sed "s/^/$1 $2 kernel: /" | logp
  (cd $EXP/probec && PYTHONPATH=src $V/bin/python -m probec.cli report $O/$1_$2.jsonl 2>&1) | grep -E "^\| vllm|^\| \?|configurations" | sed "s/^/$1 $2 /" | logp
}
for ver in 0.23.0 0.24.0; do
  [ $(left) -lt 25 ] && { log "skip vllm $ver: $(left) min left"; continue; }
  install $ver
  run $ver 8b_fp8_ckpt dense8b-fp8 default/prefix
  run $ver 8b_fp8_online dense8b default/fp8_online
  run $ver 06b_fp8_ckpt dense-fp8 default/prefix
  rm -rf $ROOT/vllm$ver
done
for ver in 0.18.1 0.19.0; do
  [ $(left) -lt 25 ] && { log "skip vllm $ver: $(left) min left"; continue; }
  install $ver
  run $ver nvfp4_unit dense-nvfp4 default/prefix
  run $ver nvfp4_modelopt dense-nvfp4-mo default/prefix
  run $ver moe_nvfp4_unit moe-nvfp4 default/prefix
  rm -rf $ROOT/vllm$ver
done
log "done ($(left) min left)"
