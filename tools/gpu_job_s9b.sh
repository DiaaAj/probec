#!/usr/bin/env bash
# GPU session 9b (RTX 3090): (1) pin the vLLM commit that fixed the MLA prefix-cache failure (bad on
# 0.25.1/0.26.0/0.27.1, exact from 0.28.0) with per-commit wheels: #50613's parent vs #50613, then the
# other MLA candidates (#51739, #51733) if needed; (2) replay #47728 (sliding window + async scheduling +
# spec decode) with the GPU n-gram drafter and decoy prompts, which make drafts get proposed and rejected.
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
# same twin as the release bisection (session 7 predates the MLA scale check)
$PY -m probec.cli compile --like $D/deepseek-ai_DeepSeek-V2-Lite.json --layers 6 --program mla --no-scale-check -o $P/mla-v2lite 2>&1 | tail -1 | tee -a out/job.log
$PY -m probec.cli compile --like $D/mistralai_Mistral-7B-v0.1.json --layers 4 -o $P/swa-mistral 2>&1 | tail -1 | tee -a out/job.log

run(){   # python probe matrix only requests max_len outfile tag
  log "$8 $(basename $2) $4"
  timeout 1200 $1 -m probec.cli run $2 --engine vllm --matrix $3 --only "$4" --requests $5 --max-len $6 --out out/$7 >> out/run.log 2>&1 \
    || echo "{\"config\": \"$4\", \"error\": \"exit/timeout\"}" >> out/$7
  python3 -c "import json; r=json.loads(open('out/$7').readlines()[-1]); print(' -> $8', r.get('program'), r.get('config'), 'gen', r.get('gen_tokens'), 'bad', r.get('mismatched_tokens'), 'bad_req', r.get('mismatched_requests'), 'fails', [(f['request'], f['prompt_len'], f['first_bad_step']) for f in r.get('failures', [])][:8], r.get('error','')[:240])" | tee -a out/job.log
}
venv(){ rm -rf /root/vc; python3 -m venv /root/vc >/dev/null 2>&1 || $PY -m venv /root/vc; /root/vc/bin/pip install -q -U pip >/dev/null 2>&1; }
wheel_url(){ curl -s "https://wheels.vllm.ai/$1/vllm/" | grep -o 'href="[^"]*x86_64[^"]*\.whl"' | head -1 \
  | sed -e 's/^href="//' -e 's/"$//' -e 's#^\.\./\.\./#https://wheels.vllm.ai/#'; }
bis(){   # sha tag: install that commit's wheel, run the failing workload, record the mismatch count
  local url; url=$(wheel_url $1)
  if [ -z "$url" ]; then log "no wheel for $2 ($1)"; echo nowheel > out/bis_$2; return; fi
  log "install $2: $url"
  venv
  /root/vc/bin/pip install -q "$url" >> out/install_$2.log 2>&1
  /root/vc/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$2.log 2>&1
  /root/vc/bin/python -c "import vllm, torch; print('VERSIONS', vllm.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  run /root/vc/bin/python $P/mla-v2lite replay default/prefix 96 3700 bisect_commits.jsonl $2
  python3 -c "import json; r=json.loads(open('out/bisect_commits.jsonl').readlines()[-1]); m=r.get('mismatched_tokens'); print('error' if m is None else m)" > out/bis_$2
  log "RESULT $2 $(cat out/bis_$2)"
}
res(){ cat out/bis_$1 2>/dev/null; }

bis e7b8d5946095a594af2cf7ca3c314b9806cb7c32 parent_of_50613
bis b38e111d3e4806a553ec2798e2b075da7a8b03d3 pr50613
if [ "$(res pr50613)" != "0" ] || [ "$(res parent_of_50613)" = "0" ]; then   # not pinned: try the later MLA fixes
  bis 65b7662d3fcb773afaf751ab29ac6960a0cf011d before_51739
  bis c76a425278bc41e9d661a377d7c166f5be42cf6b pr51739
  bis 608c12473f3f93b5452ac3c4fa13fb280b7da674 pr51733
fi

for ver in 0.25.1 0.26.0; do
  log "install vllm==$ver"
  venv
  /root/vc/bin/pip install -q "vllm==$ver" >> out/install_$ver.log 2>&1
  /root/vc/bin/pip install -q --no-deps -e /root/exp/probec >> out/install_$ver.log 2>&1
  run /root/vc/bin/python $P/swa-mistral replay default/ngram_gpu_async_decoys 96 12000 replay47728.jsonl v$ver
  run /root/vc/bin/python $P/swa-mistral replay default/ngram_spec_decoys 96 12000 replay47728.jsonl v$ver
done
rm -rf /root/vc
log "done"
