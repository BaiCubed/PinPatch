#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# R1 demo-augmented ReAct (pre-registered PROGRESS.md 2026-11-15). Qwen2.5-7B serial, max-len 16384, GPU $1 port 8171.
R=$PINPATCH_ROOT; cd $R; source env.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
G=${1:-7}; PORT=8171; S=pp-vllmS8171
tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_MAXLEN=16384 VLLM_UTIL=0.2 VLLM_KV=1610612736 bash scripts/vllm.sh $G $PORT serial > runs/vllm/r1_$PORT.log 2>&1"
  echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu$G-port$PORT (R1 fewshot ReAct) running" >> runs/PIDS.txt; }
until curl -s localhost:$PORT/v1/models | grep -q Qwen; do sleep 20; tmux has-session -t $S 2>/dev/null || { echo SERVER_FAIL; exit 1; }; done; echo SERVER_UP
cd code/eval; U=http://127.0.0.1:$PORT/v1
for e in textcraft alfworld sciworld webshop; do st=30; [ $e = textcraft ] && st=40; [ $e = webshop ] && st=10
  $T run_eval.py --env $e --split test --method react_fewshot --max-steps $st --workers 4 --run-id r1_fewshot/${e}_test --llm-url $U 2>&1 | grep -E 'SUMMARY|Traceback' &
done
FEWSHOT_SRC=pinpatch $T run_eval.py --env sciworld --split test --method react_fewshot --max-steps 30 --workers 4 --run-id r1_fewshot/sciworld_pp_test --llm-url $U 2>&1 | grep -E 'SUMMARY|Traceback' &
wait; echo R1_DONE
tmux kill-session -t $S; echo "$(date +%F_%T) - $S stopped (R1_DONE)" >> $R/runs/PIDS.txt
