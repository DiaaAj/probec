#!/usr/bin/env bash
# GPU session 15 (RTX 3090): does the SGLang NGRAM corruption (radix cache + overlap scheduler + CUDA graphs,
# concurrent shared-prefix requests) reproduce on SGLang main? No release followed 0.5.21, so main is pinned
# to a commit; 0.5.21 runs on the same machine and workload as the control.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
SHA=d0a098b2e08c080bd735cdee97479f9f39148a81
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like /root/exp/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log

repro(){   # python tag
  $1 -c "import sglang, importlib.metadata as m; print('VERSIONS sglang', sglang.__version__, 'kernel', m.version('sglang-kernel') if any(d.metadata['Name'] == 'sglang-kernel' for d in m.distributions()) else m.version('sgl-kernel'), 'torch', m.version('torch'))" 2>&1 | tail -1 | tee -a out/job.log
  log "repro $2"
  (cd /root/exp/probec && timeout 2400 $1 tools/repro_sglang_ngram.py $P/twin 2>&1) | grep -v "it/s" > out/repro_$2.log
  grep -E "NGRAM vs|request [0-9]+:|Error|error" out/repro_$2.log | head -14 | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 3
}

log "SGLang main @ $SHA"
python3 -m venv /root/sglmain >/dev/null 2>&1 || $PY -m venv /root/sglmain
/root/sglmain/bin/pip install -q -U pip >/dev/null 2>&1
git clone -q https://github.com/sgl-project/sglang /root/sglang && git -C /root/sglang checkout -q $SHA
/root/sglmain/bin/pip install -q -e "/root/sglang/python[all]" >> out/install_main.log 2>&1 \
  || /root/sglmain/bin/pip install -q -e "/root/sglang/python" >> out/install_main.log 2>&1 \
  || log "main install failed (see install_main.log)"
/root/sglmain/bin/python -c "import sglang" 2>/dev/null && repro /root/sglmain/bin/python main
rm -rf /root/sglmain

log "SGLang 0.5.21 (control)"
python3 -m venv /root/sgl0521 >/dev/null 2>&1 || $PY -m venv /root/sgl0521
/root/sgl0521/bin/pip install -q -U pip >/dev/null 2>&1
/root/sgl0521/bin/pip install -q "sglang[all]==0.5.21" >> out/install_0521.log 2>&1 \
  || /root/sgl0521/bin/pip install -q "sglang==0.5.21" >> out/install_0521.log 2>&1
repro /root/sgl0521/bin/python v0.5.21
log "done"
