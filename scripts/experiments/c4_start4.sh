#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; cd $R
for gp in "4 8133" "6 8134" "7 8135" "5 8136"; do set -- $gp
  s=pp-vllmS$2
  tmux new-session -d -s $s "VLLM_UTIL=0.2 VLLM_KV=1610612736 bash scripts/vllm.sh $1 $2 serial > runs/vllm/c4_$2.log 2>&1"
  sleep 2; pid=$(tmux list-panes -t $s -F '#{pane_pid}')
  echo "$(date +%F_%T) $pid $s vllm-serial-gpu$1-port$2 (C4 larger edit space) running" >> runs/PIDS.txt
  sleep 40
done; echo STARTED
