#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# R2(a) with phi-4 as the diagnoser (pre-registered PROGRESS.md R2: "phi-4, run when the phi-4 server is free").
# Same deterministic serial server settings as P0-2 (GPU $1, port 8140). Marker R2A_PHI4_DONE; kills its server at the end.
R=$PINPATCH_ROOT; cd $R; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
G=${1:-6}; PORT=8140; M14=$R/models/phi-4; S=pp-vllmphi4
tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_MODEL=$M14 VLLM_UTIL=0.38 VLLM_KV=2147483648 bash scripts/vllm.sh $G $PORT serial > runs/vllm/phi4_r2a.log 2>&1"
  echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu$G-port$PORT (R2a phi-4 diagnoser) running" >> runs/PIDS.txt; }
for k in $(seq 120); do curl -s localhost:$PORT/v1/models | grep -q phi-4 && break; sleep 10; done
curl -s localhost:$PORT/v1/models | grep -q phi-4 || { echo SERVER_FAIL; exit 1; }; echo SERVER_UP
cd code/eval; LLM_MODEL=$M14 $T r2a_diag.py http://127.0.0.1:$PORT/v1 phi4 2>&1 | tail -20; echo R2A_PHI4_DONE
tmux kill-session -t $S; echo "$(date +%F_%T) - $S stopped (R2A_PHI4_DONE)" >> $R/runs/PIDS.txt
