#!/usr/bin/env bash
# GPU session 2 (single GPU): positional twin (Qwen3-0.6B, real rope_theta), MoE twin (Qwen3-30B-A3B shape,
# 4 layers), MLA twins (DeepSeek-V2-Lite and DeepSeek-V3 shapes, all-dense MLPs) on vLLM and SGLang.
# Runs in /root/exp on a vast.ai box (vastai/vllm image); out/ is pulled back.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
[ -z "$PY" ] && { log "no vLLM python found"; exit 1; }
$PY -c "import vllm, torch; print('VERSIONS vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
PROBEC="$PY -m probec.cli"
D=/root/exp/probec/tests/data
P=/root/probes
mkdir -p $P

log "compile probes"
$PROBEC compile --like $D/Qwen_Qwen3-0.6B.json -o $P/recall-sc-qwen3-0.6b 2>&1 | tail -2 | tee -a out/job.log
$PROBEC compile --like $D/Qwen_Qwen3-0.6B.json --program posrecall -o $P/pos-qwen3-0.6b 2>&1 | tail -2 | tee -a out/job.log
$PROBEC compile --like $D/Qwen_Qwen3-30B-A3B.json --layers 4 --program moe -o $P/moe-qwen3-30b-a3b-L4 2>&1 | tail -2 | tee -a out/job.log
$PROBEC compile --like $D/deepseek-ai_DeepSeek-V2-Lite.json --layers 6 --program mla -o $P/mla-dsv2lite-L6 2>&1 | tail -2 | tee -a out/job.log
$PROBEC compile --like $D/deepseek-ai_DeepSeek-V3.json --layers 3 --program mla -o $P/mla-dsv3-L3 2>&1 | tail -2 | tee -a out/job.log
for pr in $P/*; do $PROBEC certify $pr > out/certify_$(basename $pr).json; echo "certify $(basename $pr) $(grep -c '"ok": false' out/certify_$(basename $pr).json) failures" | tee -a out/job.log; done

run_matrix(){   # engine python probe matrix only-list(optional) requests
  local eng=$1 py=$2 probe=$3 mat=$4 req=$5
  local tag=$(basename $probe)
  $py -m probec.cli run $probe --engine $eng --matrix $mat --list > out/plan_${eng}_${tag}.txt 2>/dev/null
  while read -r cfg; do
    log "$eng $tag $cfg"
    timeout 900 $py -m probec.cli run $probe --engine $eng --matrix $mat --only "$cfg" --requests $req \
        --out out/${eng}.jsonl >> out/${eng}_run.log 2>&1 \
      || echo "{\"config\": \"$cfg\", \"engine\": \"$eng\", \"probe\": \"$tag\", \"error\": \"exit/timeout\"}" >> out/${eng}.jsonl
    tail -1 out/${eng}.jsonl | cut -c1-300 | tee -a out/job.log
    pkill -f "sglang" >/dev/null 2>&1; sleep 1
  done < out/plan_${eng}_${tag}.txt
}

log "vLLM"
run_matrix vllm "$PY" $P/recall-sc-qwen3-0.6b quick 128
run_matrix vllm "$PY" $P/pos-qwen3-0.6b medium 128
run_matrix vllm "$PY" $P/moe-qwen3-30b-a3b-L4 quick 128
run_matrix vllm "$PY" $P/mla-dsv2lite-L6 default 128
run_matrix vllm "$PY" $P/mla-dsv3-L3 quick 64

log "SGLang install"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 \
  || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
/root/sgl/bin/python -c "import sglang, torch; print('VERSIONS sglang', sglang.__version__, 'torch', torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
run_matrix sglang /root/sgl/bin/python $P/recall-sc-qwen3-0.6b quick 128
run_matrix sglang /root/sgl/bin/python $P/pos-qwen3-0.6b quick 128
run_matrix sglang /root/sgl/bin/python $P/moe-qwen3-30b-a3b-L4 quick 128
run_matrix sglang /root/sgl/bin/python $P/mla-dsv2lite-L6 quick 128

$PY - <<'PYEOF' | tee -a out/job.log
import json, glob
for fn in sorted(glob.glob("out/*.jsonl")):
    rs = [json.loads(l) for l in open(fn)]
    print(fn, "configs", len(rs), "gen tokens", sum(r.get("gen_tokens", 0) for r in rs),
          "mismatched", sum(r.get("mismatched_tokens", 0) for r in rs),
          "errors", [(r.get("program"), r["config"]) for r in rs if "error" in r])
PYEOF
log "done"
