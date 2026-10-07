#!/usr/bin/env bash
# M1 GPU session: compile a Qwen3-0.6B probe twin, run the vLLM and SGLang matrices on it, and the
# targeted KV fault campaign on vLLM. Runs inside /root/exp on a vast.ai box (vastai/vllm image);
# everything written to out/ is pulled back.
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
PROBEC="$PY -m probec.cli"

TWIN=/root/probes/qwen3-0.6b-twin
log "compile twin"
$PROBEC compile --like /root/exp/probec/tests/data/Qwen_Qwen3-0.6B.json -o $TWIN 2>&1 | tee -a out/job.log
$PROBEC certify $TWIN > out/certify_twin.json; grep '"ok"' out/certify_twin.json | tail -1 | tee -a out/job.log

log "vLLM matrix"
$PROBEC run $TWIN --engine vllm --matrix default --list > out/vllm_plan.txt
while read -r cfg; do
  log "vllm $cfg"
  timeout 900 $PROBEC run $TWIN --engine vllm --only "$cfg" --out out/vllm.jsonl >> out/vllm_run.log 2>&1 \
    || echo "{\"config\": \"$cfg\", \"engine\": \"vllm\", \"error\": \"exit/timeout\"}" >> out/vllm.jsonl
  tail -1 out/vllm.jsonl | cut -c1-400 | tee -a out/job.log
done < out/vllm_plan.txt

log "SGLang install"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 \
  || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
/root/sgl/bin/python -c "import sglang, torch; print('VERSIONS sglang', sglang.__version__, 'torch', torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
SPROBEC="/root/sgl/bin/python -m probec.cli"
$SPROBEC run $TWIN --engine sglang --matrix default --list > out/sglang_plan.txt
while read -r cfg; do
  log "sglang $cfg"
  timeout 900 $SPROBEC run $TWIN --engine sglang --only "$cfg" --out out/sglang.jsonl >> out/sglang_run.log 2>&1 \
    || echo "{\"config\": \"$cfg\", \"engine\": \"sglang\", \"error\": \"exit/timeout\"}" >> out/sglang.jsonl
  tail -1 out/sglang.jsonl | cut -c1-400 | tee -a out/job.log
  pkill -f "sglang" >/dev/null 2>&1; sleep 2
done < out/sglang_plan.txt

$PY - <<'PYEOF' | tee -a out/job.log
import json, glob
for fn in sorted(glob.glob("out/*.jsonl")):
    rs = [json.loads(l) for l in open(fn)]
    print(fn, "configs", len(rs), "gen tokens", sum(r.get("gen_tokens", 0) for r in rs),
          "mismatched", sum(r.get("mismatched_tokens", 0) for r in rs),
          "errors", [r["config"] for r in rs if "error" in r])
PYEOF
log "done"
