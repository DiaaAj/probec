#!/usr/bin/env bash
# GPU session 16 (RTX 3090): does the SGLang NGRAM corruption reproduce on 0.5.22, the first release after
# 0.5.21? 0.5.22 (2026-10-16) contains main @ d0a098b, which was clean in session 15b. 0.5.21 runs on the
# same machine as the control; then SGLang's default matrix on 0.5.22.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/job.log; }
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
python3 --version 2>&1 | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like /root/exp/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log

install(){   # venv version
  python3 -m venv $1 >/dev/null 2>&1 || $PY -m venv $1
  $1/bin/pip install -q -U pip >/dev/null 2>&1
  $1/bin/pip install -q "sglang[all]==$2" >> out/install_$2.log 2>&1 \
    || $1/bin/pip install -q "sglang==$2" >> out/install_$2.log 2>&1 \
    || log "install $2 failed (see install_$2.log)"
}

repro(){   # python tag
  $1 -c "import sglang, importlib.metadata as m; print('VERSIONS sglang', sglang.__version__, 'kernel', m.version('sglang-kernel') if any(d.metadata['Name'] == 'sglang-kernel' for d in m.distributions()) else m.version('sgl-kernel'), 'torch', m.version('torch'), 'flashinfer', m.version('flashinfer-python'))" 2>&1 | tail -1 | tee -a out/job.log
  log "repro $2"
  (cd /root/exp/probec && timeout 900 $1 tools/repro_sglang_ngram.py $P/twin 2>&1) | grep -v "it/s" > out/repro_$2.log
  grep -E "NGRAM vs|request [0-9]+:|Error|error" out/repro_$2.log | head -14 | tee -a out/job.log
  pkill -f sglang >/dev/null 2>&1; sleep 3
}

log "SGLang 0.5.22"
install /root/sgl0522 0.5.22
/root/sgl0522/bin/python -c "import sglang" 2>/dev/null && repro /root/sgl0522/bin/python v0.5.22

log "SGLang 0.5.21 (control)"
install /root/sgl0521 0.5.21
/root/sgl0521/bin/python -c "import sglang" 2>/dev/null && repro /root/sgl0521/bin/python v0.5.21
rm -rf /root/sgl0521

log "default matrix on SGLang 0.5.22"
(cd /root/exp/probec && PYTHONPATH=src timeout 1800 /root/sgl0522/bin/python -m probec.cli run $P/twin --engine sglang \
   --matrix default --out /root/exp/out/results_0522.jsonl 2>&1) | grep -v "it/s" > out/matrix_0522.log
(cd /root/exp/probec && PYTHONPATH=src /root/sgl0522/bin/python -m probec.cli report /root/exp/out/results_0522.jsonl 2>&1) \
  | tee out/report_0522.md | tee -a out/job.log
log "done"
