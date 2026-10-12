#!/usr/bin/env bash
# GPU session 15b (RTX 3090): the SGLang NGRAM repro on SGLang main, built with its Rust extensions (main's
# radix tree is a Rust crate, so building without them would test different code). Session 15 ran the
# 0.5.21 control; its main install failed because the image has no Rust toolchain.
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

log "Rust toolchain"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >> out/rustup.log 2>&1
. "$HOME/.cargo/env"
git clone -q https://github.com/sgl-project/sglang /root/sglang && git -C /root/sglang checkout -q $SHA
(cd /root/sglang/rust && rustup show active-toolchain >> /root/exp/out/rustup.log 2>&1 || rustup toolchain install >> /root/exp/out/rustup.log 2>&1)
cargo --version 2>&1 | tee -a out/job.log
log "SGLang main @ $SHA (with Rust extensions)"
python3 -m venv /root/sglmain >/dev/null 2>&1 || $PY -m venv /root/sglmain
/root/sglmain/bin/pip install -q -U pip >/dev/null 2>&1
/root/sglmain/bin/pip install -e "/root/sglang/python[all]" >> out/install_main.log 2>&1 \
  || /root/sglmain/bin/pip install -e "/root/sglang/python" >> out/install_main.log 2>&1 \
  || log "main install failed (see install_main.log)"
/root/sglmain/bin/pip list 2>/dev/null | grep -i -E "sglang|flashinfer|^torch " | tee -a out/job.log
if /root/sglmain/bin/python -c "import sglang" 2>/dev/null; then
  /root/sglmain/bin/python -c "import sglang; print('VERSIONS sglang', sglang.__version__)" 2>&1 | tail -1 | tee -a out/job.log
  log "repro main"
  (cd /root/exp/probec && timeout 2400 /root/sglmain/bin/python tools/repro_sglang_ngram.py $P/twin 2>&1) | grep -v "it/s" > out/repro_main.log
  grep -E "NGRAM vs|request [0-9]+:|Error|error" out/repro_main.log | head -14 | tee -a out/job.log
fi
log "done"
