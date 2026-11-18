#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
cd $PINPATCH_ROOT
P=(8123 8124 8125 8126 8127 8128); i=0
go() { s=$1; shift; tmux new -d -s $s "bash scripts/ma_run.sh $* > runs/logs/$s.log 2>&1"; sleep 0.3
  echo "$(date +%F_%T) $(tmux list-panes -t $s -F "#{pane_pid}") $s ma-baseline $* running" >> runs/PIDS.txt; }
for m in ace dc; do for e in textcraft alfworld sciworld webshop; do go pp-ma-$m-$e $m $e test ${P[$((i%6))]} -; i=$((i+1)); done; done
for e in textcraft alfworld sciworld webshop; do for s in 0 1 2; do go pp-ma-reflact-$e-$s reflact $e test ${P[$((i%6))]} $s/3; i=$((i+1)); done; done
