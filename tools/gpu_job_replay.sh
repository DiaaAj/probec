#!/usr/bin/env bash
# GPU session 4: replay real fixed vLLM bugs with probes (buggy release vs fixed release):
#   vLLM #42642 (MLA prefill V unpadding; DeepSeek-V2; context split into chunks)  0.25.1 -> 0.26.0
#   vLLM #47728 (sliding-window blocks freed early with async scheduling + spec decode)  0.25.1 -> 0.26.0
# plus the targeted KV localization retest with landing / cache-hit diagnostics on the image's vLLM 0.31.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
D=/root/exp/probec/tests/data; P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V2-Lite.json --layers 6 --program mla -o $P/mla-v2lite 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/mistralai_Mistral-7B-v0.1.json --layers 4 -o $P/swa-mistral 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log

log "localization retest (vLLM 0.31)"
timeout 1200 $PY -m probec.cli run $P/twin --engine vllm --only FLASH_ATTN/default --out out/loc31.jsonl >> out/loc.log 2>&1
python3 -c "import json; r=json.loads(open('out/loc31.jsonl').readlines()[-1]); print('loc31', json.dumps(r.get('targeted_faults', r.get('targeted_faults_error'))))" | tee -a out/job.log

for ver in 0.25.1 0.26.0; do
  V=/root/v${ver//./}
  log "install vllm==$ver"
  python3 -m venv $V >/dev/null 2>&1 || $PY -m venv $V
  $V/bin/pip install -q -U pip >/dev/null 2>&1
  $V/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  $V/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  $V/bin/python -c "import vllm, torch; print('VERSIONS', vllm.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  for spec in "mla-v2lite default/shared_prefix 33 3700" "mla-v2lite default/prefix 96 3700" "swa-mistral default/ngram_spec 96 12000" "swa-mistral default/prefix 96 12000"; do
    set -- $spec
    log "vllm $ver $1 $2"
    timeout 1200 $V/bin/python -m probec.cli run $P/$1 --engine vllm --matrix replay --only "$2" --requests $3 --max-len $4 \
        --out out/replay_$ver.jsonl >> out/replay_$ver.log 2>&1 \
      || echo "{\"config\": \"$2\", \"probe\": \"$1\", \"engine\": \"vllm-$ver\", \"error\": \"exit/timeout\"}" >> out/replay_$ver.jsonl
    python3 -c "import json; r=json.loads(open('out/replay_$ver.jsonl').readlines()[-1]); print('$ver $1', r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), r.get('error','')[:200])" | tee -a out/job.log
  done
done
log "done"
