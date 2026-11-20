#!/bin/bash
# Launch library-agent scripts (rl_library/READY/*.sh, header "# MEM_GB=<n>") as TOP priority when a GPU of this host
# has >= n+4 GB free (two checks 20 s apart). The script is renamed to .launched FIRST and that file is run (bash reads
# scripts lazily), from rl_library/ with the GPU pinned. A launch counts only if the process is alive 90 s later
# (otherwise the script is restored to .sh). After a launch the GPU is skipped for 5 min (the new job is still
# allocating). HOLD (rl_wiring/HOLD) is lifted after 2 verified launches on this host, or at 06:00 UTC.
GPUS="$1"; LIB="$2"; WIRING="$3"
cd "$LIB"; mkdir -p READY; L=READY/poller.log; n_ok=0
declare -A busy_until
echo "$(date -u +%H:%M:%S) poller v2 start gpus=[$GPUS]" >> $L
freemem () { python3 "$WIRING/effree.py" $1 "$WIRING"; }      # free minus not-yet-allocated library reservations
while true; do
  for s in $(ls READY/*.sh 2>/dev/null | sort); do
    need=$(grep -m1 -oE 'MEM_GB=[0-9]+' "$s" | cut -d= -f2); need=${need:-40}; needm=$(( (need + 4) * 1024 ))
    tot=$(nvidia-smi -i ${GPUS%% *} --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$needm" -gt "$tot" ] && ! grep -q "CANNOT FIT $s" $L; then
      echo "$(date -u +%H:%M:%S) CANNOT FIT $s: MEM_GB=$need (+4) > GPU total ${tot} MiB on this host" >> $L; fi
    for g in $GPUS; do
      now=$(date +%s); [ "${busy_until[$g]:-0}" -gt "$now" ] && continue
      f1=$(freemem $g); [ "$f1" -ge "$needm" ] || continue
      sleep 20; f2=$(freemem $g); [ "$f2" -ge "$needm" ] || continue
      b=$(basename "$s" .sh); mv "$s" "$s.launched"
      ( CUDA_VISIBLE_DEVICES=$g setsid nohup bash "$s.launched" > "READY/$b.log" 2>&1 < /dev/null & )
      busy_until[$g]=$(( $(date +%s) + 300 ))
      echo "$(date -u +%H:%M:%S) LAUNCH $b on GPU $g (MEM_GB=$need, free $f2 MiB)" >> $L
      sleep 90
      if pgrep -f "$b.sh.launched" > /dev/null; then
        n_ok=$((n_ok + 1)); echo "$(date -u +%H:%M:%S) VERIFIED $b alive" >> $L
        lp=$(pgrep -f "bash $s.launched" | head -1); echo "$lp $need" >> "$WIRING/RESERVE_g$g"   # reservation for effree.py
        [ $n_ok -ge 2 ] && [ -f "$WIRING/HOLD" ] && rm -f "$WIRING/HOLD" && echo "$(date -u +%H:%M:%S) HOLD lifted (2 verified)" >> $L
      else
        echo "$(date -u +%H:%M:%S) FAILED $b (not running after 90 s; restored .sh; see READY/$b.log)" >> $L
        mv "$s.launched" "$s"; mv "READY/$b.log" "READY/$b.failed.$(date +%H%M%S).log"; busy_until[$g]=0
      fi
      break
    done
  done
  if [ -f "$WIRING/HOLD" ] && [ "$(date -u +%H%M)" -ge 0600 ] && [ "$(date -u +%H%M)" -lt 1400 ] && [ ! -f "$WIRING/HOLD.manual" ]; then
    rm -f "$WIRING/HOLD"; echo "$(date -u +%H:%M:%S) HOLD lifted (06:00 UTC deadline)" >> $L; fi
  for rf in "$WIRING"/RESERVE_g*; do [ -f "$rf" ] || continue           # drop reservations of finished library runs
    awk '{ if (system("test -d /proc/" $1) == 0) print }' "$rf" > "$rf.tmp"; mv "$rf.tmp" "$rf"; [ -s "$rf" ] || rm -f "$rf"; done
  sleep 30
done
