#!/usr/bin/env bash
# Frugal one-shot vast.ai runner: rent -> wait for ssh -> push dir -> run -> pull results -> ALWAYS destroy.
# usage: vast_run.sh <offer_id> <image> <local_exp_dir> <remote_cmd> [max_minutes] [disk_gb]
# - local_exp_dir is rsynced to /root/exp on the box; results must be written to /root/exp/out
# - hard wall-clock cap: the instance is destroyed after max_minutes no matter what
# The body is one function, parsed completely before it runs: editing this file while sessions are
# running cannot change what they execute (bash otherwise reads scripts incrementally).
main() {
  set -uo pipefail
  OFFER="$1"; IMAGE="$2"; LDIR="$3"; RCMD="$4"; MAXMIN="${5:-30}"; DISK="${6:-40}"
  LOG="$LDIR/out/vast_run.log"; mkdir -p "$LDIR/out"
  SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -i $HOME/.ssh/id_rsa"
  ts(){ date -u +%H:%M:%S; }
  echo "[$(ts)] creating instance from offer $OFFER image $IMAGE disk ${DISK}G cap ${MAXMIN}m" | tee -a "$LOG"
  CREATE=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --label probec --raw 2>&1)
  echo "$CREATE" >> "$LOG"
  IID=$(echo "$CREATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_contract',''))" 2>/dev/null)
  if [ -z "$IID" ]; then echo "create failed: $CREATE" | tee -a "$LOG"; exit 1; fi
  echo "[$(ts)] instance $IID" | tee -a "$LOG"
  START=$(date +%s)
  cleanup(){ echo "[$(ts)] destroying $IID" | tee -a "$LOG"; vastai destroy instance "$IID" -y >>"$LOG" 2>&1 || vastai destroy instance "$IID" >>"$LOG" 2>&1; 
    E=$(( ($(date +%s)-START) )); echo "[$(ts)] wall seconds billed-ish: $E" | tee -a "$LOG"; }
  trap cleanup EXIT INT TERM
  # watchdog: hard cap
  ( sleep $((MAXMIN*60)); echo "[$(ts)] WATCHDOG cap hit" >> "$LOG"; vastai destroy instance "$IID" -y >/dev/null 2>&1 || vastai destroy instance "$IID" >/dev/null 2>&1 ) &
  WD=$!
  # wait for ssh
  for i in $(seq 1 150); do
    INFO=$(vastai show instance "$IID" --raw 2>/dev/null)
    ST=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status'))" 2>/dev/null)
    HOST=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); p=(d.get('ports') or {}).get('22/tcp'); print(d.get('public_ipaddr','').strip() if p else d.get('ssh_host',''))" 2>/dev/null)
    PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); p=(d.get('ports') or {}).get('22/tcp'); print(p[0]['HostPort'] if p else d.get('ssh_port',''))" 2>/dev/null)
    if [ "$ST" = "running" ] && [ -n "$HOST" ] && ssh $SSHO -p "$PORT" root@"$HOST" true 2>/dev/null; then break; fi
    # a host that cannot start the container parks it as stopped (seen on a B200): give up after 3 minutes
    CS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cur_state'))" 2>/dev/null)
    if [ "$CS" = "stopped" ] && [ $(( $(date +%s) - START )) -gt 180 ]; then ST="stopped"; break; fi
    sleep 10
  done
  echo "[$(ts)] status=$ST host=$HOST port=$PORT" | tee -a "$LOG"
  if [ "$ST" != "running" ]; then echo "never became ready" | tee -a "$LOG"; kill $WD 2>/dev/null; exit 2; fi
  # reject hosts whose GPU memory is already taken (seen: 24/24 GB held by an invisible process)
  USED=$(ssh $SSHO -p "$PORT" root@"$HOST" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | sort -n | tail -1 | tr -d ' ')
  echo "[$(ts)] GPU memory in use before the job: ${USED:-?} MiB" | tee -a "$LOG"
  if [ -n "$USED" ] && [ "$USED" -gt 2048 ] 2>/dev/null; then
    echo "[$(ts)] GPU memory already taken: rejecting this host" | tee -a "$LOG"; kill $WD 2>/dev/null; exit 3
  fi
  ssh $SSHO -p "$PORT" root@"$HOST" "mkdir -p /root/exp" >>"$LOG" 2>&1
  tar -C "$LDIR" --exclude=out -czf - . | ssh $SSHO -p "$PORT" root@"$HOST" "tar -C /root/exp -xzf -" >>"$LOG" 2>&1
  ssh $SSHO -p "$PORT" root@"$HOST" "mkdir -p /root/exp/out && cd /root/exp && timeout $((MAXMIN*60-120)) bash -lc '$RCMD'" 2>&1 | tee -a "$LOG"
  ssh $SSHO -p "$PORT" root@"$HOST" "tar -C /root/exp/out -czf - ." | tar -C "$LDIR/out" -xzf - >>"$LOG" 2>&1
  kill $WD 2>/dev/null
  echo "[$(ts)] done" | tee -a "$LOG"
}
main "$@"
exit $?
