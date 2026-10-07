#!/usr/bin/env bash
# B200 session 14: vLLM #52796 (FlashInfer / TRT-LLM ragged MLA prefill return base-2 LSE, merged as natural
# log after a prefix hit; 0.28.0 buggy -> 0.29.0 fixed). 0.28.0 picks TRT-LLM ragged by default only for MLA
# head dims 192/64/256 (GLM-5 style), so: a DeepSeek-V3-class twin with those dims (default selection), and
# the DeepSeek-V3 twin with the prefill backend forced. Split-marker prompts give the calibrated scale-check
# head one marker in the cached context and one in the new suffix: a wrong merge moves its 4:1 ratio.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
python3 - <<'PYEOF'
import json
c = json.load(open("/root/exp/probec/tests/data/deepseek-ai_DeepSeek-V3.json"))
c.update(qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256)
json.dump(c, open("/root/probes/glmdim.json", "w"))
PYEOF
$PY -m probec.cli compile --like $P/glmdim.json --layers 3 --program mla -o $P/mla-glmdim 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V3.json --layers 3 --program mla -o $P/mla-v3 2>&1 | tail -1 | tee -a out/job.log
run(){   # python probe only requests outfile tag
  log "$6 $(basename $2) $3"
  timeout 1500 $1 -m probec.cli run $2 --engine vllm --matrix replay --only "$3" --requests $4 --max-len 8000 --out out/$5 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$3\", \"probe\": \"$(basename $2)\", \"error\": \"exit/timeout\"}" >> out/$5
  python3 -c "import json; r=json.loads(open('out/$5').readlines()[-1]); print(' -> $6', r.get('engine_version'), r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), r.get('diagnoses'), [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:6], r.get('error','')[:240])" | tee -a out/job.log
}
suite(){   # python tag
  run $1 $P/mla-glmdim default/split_markers 64 r52796.jsonl $2
  run $1 $P/mla-v3 default/split_markers_trtllm_ragged 64 r52796.jsonl $2
  run $1 $P/mla-v3 default/split_markers_flashinfer_prefill 64 r52796.jsonl $2
}
for ver in 0.28.0 0.29.0; do
  V=/root/v${ver//./}
  log "install vllm==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  suite $V/bin/python v$ver
  rm -rf $V
done
suite "$PY" v0.31
grep -h "mla_prefill\|MLA prefill\|prefill backend" out/run.log | sort | uniq -c | head -12 | tee -a out/job.log
log "done"
