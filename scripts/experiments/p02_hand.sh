#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# phi-4, ScienceWorld test, main-table config: hand guard + gate-evolved library (as abl/A1_shield_hand_test on Qwen).
# Shares the phi-4 server on GPU6/8140; restarts it if p02_resume.sh (or the guard) stopped it; run_eval resumes.
R=$PINPATCH_ROOT; cd $R; source env.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
M=$R/models/phi-4; PORT=8140; S=pp-vllmphi4; O=p02_phi4/sci_ourshand_test
for k in 1 2 3 4 5; do
  if ! curl -s localhost:$PORT/v1/models | grep -q phi-4; then
    tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_MODEL=$M VLLM_UTIL=0.38 VLLM_KV=2147483648 bash scripts/vllm.sh 6 $PORT serial > runs/vllm/phi4_hand.log 2>&1"
      echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu6-port$PORT (phi-4 hand-only) running" >> runs/PIDS.txt; }
    until curl -s localhost:$PORT/v1/models | grep -q phi-4; do sleep 20; tmux has-session -t $S 2>/dev/null || break; done
  fi
  ( cd code/eval_frozen_c3r2; LLM_MODEL=$M PATCHES=$R/code/eval_frozen_c3r2/lib_gate_r2.json META_SHIELD=hand $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id $O --llm-url http://127.0.0.1:$PORT/v1 2>&1 | grep -E 'SUMMARY|Traceback' )
  [ $(wc -l < runs/$O/episodes.jsonl) -ge 200 ] && break
done; echo P02H_DONE
until grep -q P02R_DONE runs/logs/p02_resume.log; do sleep 120; done
tmux kill-session -t $S 2>/dev/null && echo "$(date +%F_%T) - $S stopped (P02H_DONE, P02R_DONE)" >> runs/PIDS.txt
