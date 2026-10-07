#!/usr/bin/env bash
# GPU session 3: investigate SGLang NGRAM mismatches on the session-1 probe; real-model NGRAM check;
# vLLM targeted KV fault localization retest (fused K/V layout fix).
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
TWIN=/root/probes/twin
$PY -m probec.cli compile --like /root/exp/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $TWIN 2>&1 | tail -1 | tee -a out/job.log

log "vLLM targeted localization"
timeout 1200 $PY -m probec.cli run $TWIN --engine vllm --only FLASH_ATTN/default --out out/vllm_loc.jsonl >> out/vllm_run.log 2>&1
tail -c 1500 out/vllm_loc.jsonl | tee -a out/job.log; echo

log "SGLang install"
python3 -m venv /root/sgl >/dev/null 2>&1 || $PY -m venv /root/sgl
/root/sgl/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl/bin/pip install -q "sglang[all]==0.5.21" >> out/sgl_install.log 2>&1 || /root/sgl/bin/pip install -q "sglang==0.5.21" >> out/sgl_install.log 2>&1
/root/sgl/bin/pip install -q --no-deps -e /root/exp/probec >> out/sgl_install.log 2>&1
for v in A_baseline H_nospec B_noradix C_nooverlap D_nocudagraph E_triton G_draft2 I_noshared_prompts F_sequential; do
  log "ngram variant $v"
  timeout 900 /root/sgl/bin/python /root/exp/probec/tools/sgl_ngram_investigate.py $TWIN $v out/sgl_ngram.jsonl >> out/sgl_ngram.log 2>&1 \
    || echo "{\"variant\": \"$v\", \"error\": \"exit/timeout\"}" >> out/sgl_ngram.jsonl
  tail -1 out/sgl_ngram.jsonl | cut -c1-300 | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 2
done
log "real model"
for m in nospec spec nospec_det spec_det; do
  timeout 900 /root/sgl/bin/python /root/exp/probec/tools/sgl_ngram_realmodel.py out/sgl_real.jsonl $m >> out/sgl_real.log 2>&1 || echo "real $m failed" | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 2
done
python3 - <<'PYEOF' | tee -a out/job.log
import json
rs = {}
for l in open("out/sgl_real.jsonl"):
    r = json.loads(l); rs[r["mode"]] = r["outputs"]
for a, b in (("nospec", "spec"), ("nospec_det", "spec_det"), ("nospec", "nospec_det")):
    if a in rs and b in rs:
        div = [(i, next((j for j, (x, y) in enumerate(zip(p, q)) if x != y), None)) for i, (p, q) in enumerate(zip(rs[a], rs[b]))]
        bad = [(i, j) for i, j in div if j is not None]
        print(f"real-model {a} vs {b}: {len(bad)}/{len(div)} requests diverge; first positions {sorted(j for _, j in bad)[:12]}")
PYEOF
log "done"
