#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P0-3: evaluate the SFT LoRA (AgentTraj-L) on the four TEST splits with the ReAct loop in its training prompt format
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
until grep -q SFT_DONE $R/scripts/logs/sft.log; do sleep 120; done
A=$R/runs/sft/lora_r16; PORT=8150; cd $R
tmux new-session -d -s pp-vllmsft "VLLM_UTIL=0.22 VLLM_KV=1610612736 VLLM_EXTRA='--enable-lora --lora-modules sft=$A --max-lora-rank 16' bash scripts/vllm.sh 3 $PORT serial > runs/vllm/p03_$PORT.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-vllmsft -F '#{pane_pid}') pp-vllmsft vllm-serial-gpu3-port$PORT (P0-3 SFT LoRA) running" >> runs/PIDS.txt
until curl -s localhost:$PORT/v1/models | grep -q '"sft"'; do sleep 20; tmux has-session -t pp-vllmsft 2>/dev/null || { echo SERVER_FAIL; tail -20 runs/vllm/p03_$PORT.log; exit 1; }; done; echo SERVER_UP
cd $R/code/eval; export LLM_MODEL=sft REACT_INTRO=agenttraj; U=http://127.0.0.1:$PORT/v1
for e in textcraft:40 alfworld:30 sciworld:30 webshop:10; do IFS=: read env st <<< "$e"
  $T run_eval.py --env $env --split test --method react --max-steps $st --workers 4 --run-id p03_sft/${env}_sft_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" &
done; wait; echo P03_DONE
