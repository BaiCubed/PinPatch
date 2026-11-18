#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P1-3 held-out task types (pre-registered PROGRESS.md 2026-11-15). Qwen2.5-7B serial on GPU $1 port 8170.
R=$PINPATCH_ROOT; cd $R; source env.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
G=${1:-7}; PORT=8170; S=pp-vllmS8170
tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_UTIL=0.2 VLLM_KV=1610612736 bash scripts/vllm.sh $G $PORT serial > runs/vllm/p13_$PORT.log 2>&1"
  echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu$G-port$PORT (P1-3 held-out types) running" >> runs/PIDS.txt; }
until curl -s localhost:$PORT/v1/models | grep -q Qwen; do sleep 20; tmux has-session -t $S 2>/dev/null || { echo SERVER_FAIL; exit 1; }; done; echo SERVER_UP
cd code/eval_frozen_c3r2_hot
for h in type family; do
  HOLD_TYPE=$h PATCHES=$R/code/eval_frozen_c3r2_hot/lib_gate_r2.json LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id p13_hot/test_hold_$h --llm-url http://127.0.0.1:$PORT/v1 2>&1 | grep -E 'SUMMARY|Traceback' &
done; wait; echo P13_DONE
tmux kill-session -t $S; echo "$(date +%F_%T) - $S stopped (P13_DONE)" >> $R/runs/PIDS.txt
