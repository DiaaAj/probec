#!/usr/bin/env bash
# GPU session 17 (RTX 3090): bisect the SGLang NGRAM bug (radix cache + overlap scheduler + CUDA graphs,
# concurrent shared-prefix requests). v0.5.21 gets 8/192 requests wrong; main @ d0a098b and 0.5.22 get none.
# v0.5.21 is a release-branch tag two cherry-picks (a test change and an InternVL fix) past main's bd78095, so
# the bisection runs on main from bd78095: 813 commits, about 10 steps. tools/bisect_ngram_step.sh builds and
# tests one commit. Both ends are tested first with the same build on this machine, and the transition the
# bisection finds is tested again at the end.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
cd /root/exp && mkdir -p out
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a /root/exp/out/job.log; }
export OLD=bd780950300a0c236509352bab44d8a8a98f36c2 NEW=d0a098b2e08c080bd735cdee97479f9f39148a81
# no bisection step starts after BUDGET_MIN minutes; the final re-test may run 20 minutes past it
export JOB_START=$(date +%s) BUDGET_MIN=120 CARGO_TARGET_DIR=/root/cargo-target
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee -a out/job.log
echo "cpus $(nproc), $(python3 --version)" | tee -a out/job.log
$PY -m pip install -q --no-deps -e /root/exp/probec 2>&1 | tail -1
export P=/root/probes; mkdir -p $P
$PY -m probec.cli compile --like /root/exp/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 | tail -1 | tee -a out/job.log

log "tools"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >> out/rustup.log 2>&1
. "$HOME/.cargo/env"
python3 -m venv /root/uvenv >> out/uv.log 2>&1 && /root/uvenv/bin/pip install -q uv >> out/uv.log 2>&1
export UV=/root/uvenv/bin/uv
echo "$(cargo --version 2>&1), $($UV --version 2>&1)" | tee -a out/job.log
git clone -q https://github.com/sgl-project/sglang /root/sglang
STEP=/root/exp/probec/tools/bisect_ngram_step.sh
cd /root/sglang

log "ends of the range"
git checkout -q $OLD && bash $STEP; e_old=$?
git checkout -q $NEW && bash $STEP; e_new=$?
if [ $e_old != 0 ] || [ $e_new != 1 ]; then
  log "the ends do not behave as before (bd78095 exit $e_old, want 0 = buggy; d0a098b exit $e_new, want 1 = clean); no bisection"
else
  log "bisect"
  git bisect start --term-new=clean --term-old=buggy $NEW $OLD 2>&1 | tee -a /root/exp/out/bisect_run.log
  git bisect run bash $STEP 2>&1 | tee -a /root/exp/out/bisect_run.log
  git bisect log > /root/exp/out/bisect_log.txt 2>&1
  # git quotes the term in newer versions ("is the first 'clean' commit"); refs/bisect/clean is the answer
  X=""
  grep -q -E "is the first '?clean'? commit" /root/exp/out/bisect_run.log && X=$(git rev-parse refs/bisect/clean)
  git bisect reset > /dev/null 2>&1
  if [ -n "$X" ]; then
    log "first clean commit: $(git log -1 --format='%h %s' $X)"
    git show --stat --format='%H%n%an%n%ad%n%n%s%n%n%b' $X > /root/exp/out/first_clean_commit.txt 2>&1
    git show $X > /root/exp/out/first_clean_commit.diff 2>&1
    log "re-test the transition"
    export BUDGET_MIN=$((BUDGET_MIN + 20))
    git checkout -q $X^ && bash $STEP
    git checkout -q $X && bash $STEP
  else
    log "the bisection did not finish (see bisect_run.log and bisect_log.txt)"
  fi
fi
sort -t$'\t' -k2,2n /root/exp/out/bisect_results.tsv 2>/dev/null | cut -f2-5 | tee -a /root/exp/out/job.log
log "done"
