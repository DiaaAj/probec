#!/usr/bin/env bash
# GPU session 20 (H100): quantized checkpoints on Hopper (M6a and M6b). `probec quantize` writes FP8
# (per-tensor; 128x128 blocks as in Qwen3-*-FP8 and DeepSeek-V3), AWQ and GPTQ (Machete on SM90) and NVFP4
# (Marlin W4A16 below SM100) copies of dense, 8B-shaped, MoE and MLA twins; vLLM 0.31 and SGLang 0.5.22 run
# them with every entry in its own process, plus online FP8 (W8A8, dynamic activations). Also reruns the
# reference model's NVFP4 and MXFP4 activation checks, which failed on a device mismatch in session 19.
# Environment: CREATED = the instance's creation time (epoch seconds) as the launcher saw it.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=/root; EXP=/root/exp; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }
CAP_MIN=125
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
compile Qwen_Qwen3-8B.json dense8b --layers 4
compile Qwen_Qwen3-30B-A3B.json moe --layers 4 --program moe
compile deepseek-ai_DeepSeek-V3.json mla --layers 3 --program mla
for t in dense moe mla; do
  $PY -m probec.cli certify $P/$t --quant fp8,fp8-block > $O/certify_$t.json 2>&1
  $PY -c "
import json; r = json.load(open('$O/certify_$t.json')); q = r['checks']['quantized_weights']['formats']
print('$t certify', {f: '%d/%d' % (v['exact'], v['tensors']) for f, v in q.items()})" 2>&1 | logp
done
for spec in dense:fp8 dense:fp8-block dense8b:fp8-block moe:fp8-block mla:fp8-block dense:int4-awq dense:int4-gptq \
            dense:nvfp4 moe:int4-awq dense8b:int4-awq; do
  t=${spec%%:*}; f=${spec#*:}
  $PY -m probec.cli quantize $P/$t --format $f --device cuda -o $P/$t-$f 2>&1 | grep -v -i warn | tail -1 | logp
done

log "reference model, NVFP4 and MXFP4 activations (rerun of session 19)"
for act in nvfp4 mxfp4; do
  (cd $EXP/probec && timeout 900 $PY -c "
import json
from probec.quant import check_quant
print(json.dumps(check_quant('$P/dense', act='$act', weight='fp8', n_prompts=24, max_len=4000, n_new=64, device='cuda')))
" 2>&1) | grep -v -i -E "warn|it/s" | tail -1 | logp
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
run v_online $PY vllm $P/dense quant
for c in dense-fp8 dense-fp8-block dense8b-fp8-block moe-fp8-block dense-int4-awq dense-int4-gptq dense-nvfp4 \
         moe-int4-awq dense8b-int4-awq; do
  run v_$c $PY vllm $P/$c quick FLASH_ATTN/default
done
run v_mla-fp8-block_flashmla $PY vllm $P/mla-fp8-block hopper FLASHMLA/default
run v_mla-fp8-block_triton $PY vllm $P/mla-fp8-block hopper TRITON_MLA/default

log "SGLang 0.5.22"
V=$ROOT/sgl0522
if [ $(left) -ge 25 ]; then
  t0=$(date +%s)
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "sglang[all]==0.5.22" >> $O/install_0.5.22.log 2>&1 || log "install failed (see install_0.5.22.log)"
  log "  install took $(( $(date +%s) - t0 )) s"
  run s_online $V/bin/python sglang $P/dense quant
  for c in dense-fp8 dense-fp8-block moe-fp8-block dense-int4-awq dense-int4-gptq dense-nvfp4 moe-int4-awq \
           dense8b-int4-awq; do
    run s_$c $V/bin/python sglang $P/$c quick flashinfer/default
  done
  run s_mla-fp8-block $V/bin/python sglang $P/mla-fp8-block quick triton/default
fi
log "done ($(left) min left)"
