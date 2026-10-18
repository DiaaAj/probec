#!/usr/bin/env bash
# GPU session 18 (RTX 3090): is the SGLang NGRAM bug fixed in 0.5.22, or only hidden? Session 17 bisected the
# fix to #42720, which skips the discarded trailing decode of the overlap scheduler only for requests that
# finish by max_new_tokens; the standard repro ends every request by length. Runs, in this order:
#   A  0.5.22, length endings (the standard repro): control, 0 wrong expected
#   B  0.5.22, stop-token endings (tools/repro_sglang_ngram.py --finish stop)
#   D  0.5.22, stop-token endings, overlap scheduler off: negative control
#   C  0.5.22 with #42720's skip switched off (one line of the installed scheduler.py), length endings
#   E  0.5.21, length endings: control, the usual 8 wrong requests expected
#   F  0.5.21, stop-token endings
#   G  0.5.22 (scheduler.py restored), length endings, SGLANG_TEST_RETRACT=1: only if time remains
# Environment: CREATED = the instance's creation time in epoch seconds as the launcher saw it (the runner's
# watchdog counts from creation, image pull included); ROOT and EXP default to /root and /root/exp.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_TELEMETRY=1
ROOT=${ROOT:-/root}; EXP=${EXP:-/root/exp}; O=$EXP/out
cd $EXP && mkdir -p $O
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $O/job.log; }
logp(){ while IFS= read -r line; do log "$line"; done; }      # timestamp every line of stdin
CAP_MIN=100                 # the runner's watchdog cap (minutes from instance creation)
RUN_TIMEOUT=1200            # seconds per repro run (two engines)
NOW=$(date +%s)
# out/ is copied back only after the job ends, so no run starts unless its timeout ends before the watchdog
if [ -n "${CREATED:-}" ] && [ "$CREATED" -le $((NOW + 300)) ] 2>/dev/null && [ $((NOW - CREATED)) -lt $((CAP_MIN * 60)) ]; then
  DEADLINE=$((CREATED + CAP_MIN * 60))
else
  DEADLINE=$((NOW + CAP_MIN * 60 - 30 * 60))      # creation time unknown: assume a 30-minute image pull
fi
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
NEED=$(( RUN_TIMEOUT / 60 + 5 ))  # minutes a run may need: its timeout, cleanup and the copy back

nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1 | logp
log "cpus $(nproc), cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo ?), $(python3 --version 2>&1)"
log "$(left) min left before the watchdog (CREATED=${CREATED:-unset})"
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python python3; do
  if $cand -c "import vllm" 2>/dev/null; then PY=$cand; break; fi
done
if [ -z "$PY" ]; then PY=python3; log "no python with vllm found; compiling the twin with python3"; fi
$PY -m pip install -q --no-deps -e $EXP/probec 2>&1 | tail -1 | logp
P=$ROOT/probes; mkdir -p $P
$PY -m probec.cli compile --like $EXP/probec/tests/data/Qwen_Qwen3-0.6B.json --no-scale-check -o $P/twin 2>&1 \
  | tail -1 | logp
if [ ! -f $P/twin/probe_meta.json ]; then log "the twin did not compile; stopping"; exit 1; fi

install(){   # venv version (pip in a python3 -m venv, as in session 16)
  local t0=$(date +%s)
  log "install sglang[all]==$2 into $1"
  python3 -m venv $1 >/dev/null 2>&1 || $PY -m venv $1
  $1/bin/pip install -q -U pip >/dev/null 2>&1
  $1/bin/pip install -q "sglang[all]==$2" >> $O/install_$2.log 2>&1 \
    || $1/bin/pip install -q "sglang==$2" >> $O/install_$2.log 2>&1 \
    || log "install $2 failed (see install_$2.log)"
  $1/bin/pip freeze > $O/freeze_$2.txt 2>&1
  log "  install of $2 took $(( $(date +%s) - t0 )) s"
}

versions(){   # venv
  $1/bin/python -c "
import importlib.metadata as m, sglang
def v(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print('VERSIONS sglang', sglang.__version__, 'kernel', v('sglang-kernel') or v('sgl-kernel'), 'torch', v('torch'),
      'flashinfer', v('flashinfer-python'), 'flashinfer-cubin', v('flashinfer-cubin'))
" 2>&1 | tail -1 | logp
}

run(){   # tag venv env-assignment-or-dash repro-args...
  local tag=$1 V=$2 envs=$3; shift 3
  [ "$envs" = "-" ] && envs=""
  if [ $(left) -lt $NEED ]; then log "skip $tag: $(left) min left before the watchdog"; return 1; fi
  log "run $tag: ${envs:+$envs }$V/bin/python tools/repro_sglang_ngram.py $P/twin $*"
  local t0=$(date +%s) used="" i
  (cd $EXP/probec && env $envs timeout $RUN_TIMEOUT $V/bin/python tools/repro_sglang_ngram.py $P/twin "$@" \
     --dump $O/dump_$tag.json 2>&1) | grep -v "it/s" > $O/repro_$tag.log
  local rc=${PIPESTATUS[0]}
  # the repro and the SGLang processes it spawned all run the venv's python
  pkill -9 -f "^$V/bin/python" > /dev/null 2>&1
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')
    [ "${used:-0}" -lt 1500 ] 2>/dev/null && break; sleep 2
  done
  log "  $tag: exit $rc after $(( $(date +%s) - t0 )) s; GPU memory after cleanup ${used:-?} MiB"
  grep -E "^(NGRAM vs|no-spec vs|stop mode|  request [0-9]+:|  NGRAM wrong)" $O/repro_$tag.log | head -40 | logp
  if ! grep -q "^NGRAM vs no-spec" $O/repro_$tag.log; then
    log "  $tag: no result; last lines of repro_$tag.log:"
    grep -v "^\s*$" $O/repro_$tag.log | tail -6 | cut -c1-240 | logp
  fi
  echo "$tag" >> $O/runs_done.txt
}

SCHED=""
patch_skip(){   # venv: switch #42720's skip off (one line of the installed scheduler.py) and check the change
  SCHED=$(find $1/lib -path "*/sglang/srt/managers/scheduler.py" 2>/dev/null | head -1)
  if [ -z "$SCHED" ]; then log "scheduler.py not found under $1/lib"; return 1; fi
  local pat='^ *self\.enable_skip_finishing_decode = True$' n
  n=$(grep -c "$pat" $SCHED)
  if [ "$n" != 1 ]; then log "expected one 'enable_skip_finishing_decode = True' line in $SCHED, found $n"; return 1; fi
  cp -p $SCHED $ROOT/scheduler.py.orig
  sed -i 's/^\( *\)self\.enable_skip_finishing_decode = True$/\1self.enable_skip_finishing_decode = False/' $SCHED
  rm -f $(dirname $SCHED)/__pycache__/scheduler.*.pyc
  local d minus plus L fn
  d=$(diff $ROOT/scheduler.py.orig $SCHED)
  log "patched $SCHED; diff against the original:"
  logp <<< "$d"
  minus=$(grep -c '^<' <<< "$d"); plus=$(grep -c '^>' <<< "$d")
  L=$(sed -n 's/^\([0-9]*\)c\([0-9]*\)$/\2/p' <<< "$d" | head -1)
  # the function the changed line belongs to: the last def above it
  fn=$(awk -v L="${L:-0}" 'NR < L && /^ *def /{d = $0} NR == L {print d}' $SCHED | sed 's/^ *//')
  log "  $minus line removed, $plus added, at line ${L:-?}, in ${fn:-?}"
  if [ "$minus" != 1 ] || [ "$plus" != 1 ] || [ -z "$L" ] || [ "$(grep -c "$pat" $SCHED)" != 0 ] \
     || [ "$(grep '^>' <<< "$d" | sed 's/^> *//')" != "self.enable_skip_finishing_decode = False" ] \
     || [[ "$fn" != "def event_loop_overlap"* ]]; then
    log "  the patch check failed; restoring the original"; restore_skip; return 1
  fi
  # what python imports must be the patched file
  timeout 180 $1/bin/python -c "
import inspect, sglang.srt.managers.scheduler as s
src = inspect.getsource(s.Scheduler.event_loop_overlap)
print('imported', s.__file__, '| event_loop_overlap sets the skip to', 'False' if 'enable_skip_finishing_decode = False' in src else 'True' if 'enable_skip_finishing_decode = True' in src else '?')
" 2>&1 | tail -1 | cut -c1-240 | logp
  return 0
}

restore_skip(){
  cp -p $ROOT/scheduler.py.orig $SCHED && rm -f $(dirname $SCHED)/__pycache__/scheduler.*.pyc
  if cmp -s $ROOT/scheduler.py.orig $SCHED; then log "restored the original scheduler.py"; return 0; fi
  log "could not restore scheduler.py"; return 1
}

V22=$ROOT/sgl0522; V21=$ROOT/sgl0521; RESTORED=0
log "SGLang 0.5.22"
install $V22 0.5.22
if $V22/bin/python -c "import sglang" 2>/dev/null; then
  versions $V22
  run A $V22 -
  run B $V22 - --finish stop
  run D $V22 - --finish stop --extra '{"disable_overlap_schedule": true}'
  if [ $(left) -lt $NEED ]; then
    log "skip C: $(left) min left before the watchdog"
  elif patch_skip $V22; then
    run C $V22 -
    restore_skip && RESTORED=1
  fi
else
  log "sglang 0.5.22 does not import (see install_0.5.22.log)"
fi

if [ $(left) -lt $((NEED + 10)) ]; then
  log "skip SGLang 0.5.21: $(left) min left before the watchdog"
else
  log "SGLang 0.5.21"
  install $V21 0.5.21
  if $V21/bin/python -c "import sglang" 2>/dev/null; then
    versions $V21
    run E $V21 -
    run F $V21 - --finish stop
  else
    log "sglang 0.5.21 does not import (see install_0.5.21.log)"
  fi
fi

if [ $RESTORED = 1 ] && [ $(left) -ge $((NEED + 5)) ]; then
  run G $V22 SGLANG_TEST_RETRACT=1
else
  log "skip G (scheduler.py restored: $RESTORED, $(left) min left)"
fi

log "summary (tag: NGRAM vs no-spec / NGRAM vs oracle / no-spec vs oracle)"
for t in A B D C E F G; do
  f=$O/repro_$t.log; [ -f $f ] || continue
  log "  $t: $(grep -E '^NGRAM vs no-spec' $f | tail -1) | $(grep -E '^no-spec vs exact' $f | tail -1)"
done
log "done"
