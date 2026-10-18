#!/usr/bin/env bash
# One `git bisect run` step for the SGLang NGRAM bug (GPU session 17): build SGLang at the checked-out commit
# from source, Rust extensions included, in a fresh venv, then run tools/repro_sglang_ngram.py.
# Exit 0: the bug reproduces (bisect term "buggy"); 1: no wrong requests ("clean"); 125: untestable (the build
# failed, the repro crashed or timed out, or the run without speculation is wrong too); 255: the job's time
# budget is used up, which stops `git bisect run`.
# Uses OLD, P (the probe dir), UV, JOB_START and BUDGET_MIN from the job script.
set -u
R=/root/sglang; O=/root/exp/out; V=/root/v
SHA=$(git -C $R rev-parse HEAD); S=${SHA:0:10}
N=$(git -C $R rev-list --count $OLD..$SHA)
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
if [ $(( $(date +%s) - JOB_START )) -gt $(( BUDGET_MIN * 60 )) ]; then
  log "time budget used up before testing $S; stopping"; exit 255
fi
log "test $S ($N/813): $(git -C $R log -1 --format=%s $SHA | cut -c1-100)"
# build from exactly this commit: no build/ or egg-info left from another one (the Cargo target dir is outside)
git -C $R clean -ffdxq
t0=$(date +%s); how=""
rm -rf $V
if $UV venv -q -p python3 $V >> $O/install_$S.log 2>&1 \
   && $UV pip install -p $V/bin/python --refresh-package sglang "sglang[all] @ file://$R/python" >> $O/install_$S.log 2>&1; then
  how=uv
else
  log "  uv install failed ($(grep -v '^\s*$' $O/install_$S.log | tail -1 | cut -c1-120)); trying pip"
  rm -rf $V; git -C $R clean -ffdxq
  python3 -m venv $V >> $O/install_$S.log 2>&1 && $V/bin/pip install -q -U pip >> $O/install_$S.log 2>&1
  $V/bin/pip install "sglang[all] @ file://$R/python" >> $O/install_$S.log 2>&1 && how=pip
fi
git -C $R reset -q --hard   # if the build touched tracked files, `git bisect` could not check out the next commit
VERS=$($V/bin/python -c "
import importlib.metadata as m, sglang
def v(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print('sglang', sglang.__version__, 'kernel', v('sglang-kernel') or v('sgl-kernel'), 'torch', v('torch'),
      'flashinfer', v('flashinfer-python'))
" 2>&1 | tail -1)
if [ -z "$how" ] || [ "${VERS%% *}" != "sglang" ]; then
  log "  install failed (${VERS:0:150}); skipping"; exit 125
fi
log "  $VERS ($how, $(( $(date +%s) - t0 )) s)"
case "$VERS" in *+g*) case "$VERS" in *+g${SHA:0:7}*) ;; *) log "  the build is not of this commit; skipping"; exit 125;; esac;; esac
t1=$(date +%s)
(cd /root/exp/probec && timeout 1500 $V/bin/python tools/repro_sglang_ngram.py $P/twin 2>&1) | grep -v "it/s" > $O/repro_$S.log
# the repro and the SGLang processes it spawned all run the step venv's python
pkill -9 -f "^$V/bin/python" > /dev/null 2>&1
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | tr -d ' ')
  [ "${used:-0}" -lt 1500 ] && break; sleep 2
done
res=$(grep -E "^NGRAM vs no-spec" $O/repro_$S.log | tail -1)
base=$(grep -E "^no-spec vs exact oracle" $O/repro_$S.log | tail -1)
wrong=$(echo "$res" | sed -n 's/.*exact oracle: \([0-9]*\) wrong.*/\1/p')
bwrong=$(echo "$base" | sed -n 's/^no-spec vs exact oracle: \([0-9]*\) wrong.*/\1/p')
if [ -z "$wrong" ] || [ -z "$bwrong" ]; then
  log "  no result ($(grep -E "Error|error" $O/repro_$S.log | grep -v thiserror | tail -1 | cut -c1-150)); skipping"
  exit 125
fi
log "  $res ($(( $(date +%s) - t1 )) s)"
log "  $base"
printf "%s\t%s\t%s\t%s\t%s\n" "$SHA" "$N" "$wrong" "$bwrong" "$VERS" >> $O/bisect_results.tsv
if [ "$bwrong" -gt 0 ]; then log "  the run without speculation is wrong too; skipping"; exit 125; fi
if [ "$wrong" -gt 0 ]; then log "  -> buggy"; exit 0; fi
log "  -> clean"; exit 1
