#!/usr/bin/env bash
# GPU session 23 (RTX 3090, a different host from session 21): SGLang 0.5.22's quantized-path failures.
#   A. NVFP4 through Marlin W4A16 garbled every request in sessions 20 and 21 (dense and MoE) while vLLM was
#      exact. Suspected: SGLang lacks vLLM #34577 (small NVFP4 block scales underflow in Marlin's BF16 scale
#      trick). probec's NVFP4 uses weight_scale_2 = 1; the same twin with ModelOpt-style scales (block
#      scales near 448) should be exact if that is the cause.
#   B. One request each went wrong mid-generation with the dense FP8-block twin and the MoE AWQ twin
#      (Marlin) in session 21, while vLLM was exact: rerun twice (deterministic?), then switch off the radix
#      cache, the overlap scheduler or CUDA graphs, or change the attention backend; bf16 control.
#   C. probec run --only on SGLang's MoE FP8 (Triton fp8e4nv fails on SM86) must now leave an error record.
# Environment: CREATED = the instance's creation time (epoch seconds) as the launcher saw it.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=/root; EXP=/root/exp; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }
CAP_MIN=100
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
compile Qwen_Qwen3-30B-A3B.json moe --layers 2 --program moe
q(){ $PY -m probec.cli quantize $P/$1 --format $2 --device cuda -o $P/$3 "${@:4}" 2>&1 | grep -v -i warn | tail -1 | logp; }
q dense nvfp4 dense-nvfp4
q dense nvfp4 dense-nvfp4-mo --nvfp4-scales modelopt
q dense fp8-block dense-fp8-block
q moe int4-awq moe-int4-awq
q moe fp8 moe-fp8

V=$ROOT/sgl0522
t0=$(date +%s)
python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
$V/bin/pip install -q -U pip >/dev/null 2>&1
$V/bin/pip install -q "sglang[all]==0.5.22" >> $O/install_0.5.22.log 2>&1 || log "install failed (see install_0.5.22.log)"
log "SGLang 0.5.22 installed in $(( $(date +%s) - t0 )) s"

variant(){   # tag twin name kwargs-json
  [ $(left) -lt 10 ] && { log "skip $1: $(left) min left"; return; }
  (cd $EXP/probec && timeout 900 $V/bin/python tools/sgl_variant.py $P/$2 $O/$1.jsonl $3 "$4" 2>&1) \
    | grep -v "it/s" > $O/$1.log
  grep -h "^VARIANT" $O/$1.log | sed "s/^/$1 /" | logp
  [ -n "$(grep -h '^VARIANT' $O/$1.log)" ] || log "$1: no result ($(grep -E 'Error|error' $O/$1.log | grep -v thiserror | tail -1 | cut -c1-160))"
  pkill -9 -f "^$V/bin/python" > /dev/null 2>&1; sleep 3
}
FI='"attention_backend": "flashinfer"'
log "A. NVFP4 (Marlin W4A16): probec's unit scales vs ModelOpt-style scales"
variant a_nvfp4_unit dense-nvfp4 flashinfer/default "{$FI}"
variant a_nvfp4_modelopt dense-nvfp4-mo flashinfer/default "{$FI}"
log "B. dense FP8-block (Marlin W8A16)"
variant b_fp8block_1 dense-fp8-block flashinfer/default "{$FI}"
variant b_fp8block_2 dense-fp8-block flashinfer/default "{$FI}"
variant b_fp8block_noradix dense-fp8-block noradix "{$FI, \"disable_radix_cache\": true}"
variant b_fp8block_nooverlap dense-fp8-block nooverlap "{$FI, \"disable_overlap_schedule\": true}"
variant b_fp8block_nograph dense-fp8-block nograph "{$FI, \"disable_cuda_graph\": true}"
variant b_fp8block_triton dense-fp8-block triton '{"attention_backend": "triton"}'
variant b_bf16_control dense flashinfer/default "{$FI}"
log "B. MoE AWQ (Marlin MoE)"
variant b_moeawq_1 moe-int4-awq flashinfer/default "{$FI}"
variant b_moeawq_2 moe-int4-awq flashinfer/default "{$FI}"
variant b_moeawq_noradix moe-int4-awq noradix "{$FI, \"disable_radix_cache\": true}"
variant b_moeawq_nooverlap moe-int4-awq nooverlap "{$FI, \"disable_overlap_schedule\": true}"
variant b_moeawq_nograph moe-int4-awq nograph "{$FI, \"disable_cuda_graph\": true}"
variant b_moe_bf16_control moe flashinfer/default "{$FI}"
log "C. probec run --only on a crashing entry (SGLang MoE FP8, Triton fp8e4nv on SM86)"
if [ $(left) -ge 10 ]; then
  (cd $EXP/probec && PYTHONPATH=src timeout 900 $V/bin/python -m probec.cli run $P/moe-fp8 --engine sglang --matrix quick \
     --only flashinfer/default --out $O/c_only_crash.jsonl 2>&1) | grep -v "it/s" > $O/c_only_crash.log
  (cd $EXP/probec && PYTHONPATH=src $V/bin/python -m probec.cli report $O/c_only_crash.jsonl 2>&1) | grep -E "^\| |configurations" | logp
fi
log "vLLM 0.31 on the two NVFP4 twins (control)"
for t in dense-nvfp4 dense-nvfp4-mo; do
  [ $(left) -lt 10 ] && break
  (cd $EXP/probec && PYTHONPATH=src timeout 900 $PY -m probec.cli run $P/$t --engine vllm --matrix quick --only FLASH_ATTN/default \
     --out $O/v_$t.jsonl 2>&1) | grep -v "it/s" > $O/v_$t.log
  (cd $EXP/probec && PYTHONPATH=src $PY -m probec.cli report $O/v_$t.jsonl 2>&1) | grep -E "^\| vllm|configurations" | sed "s/^/$t /" | logp
done
log "done ($(left) min left)"
