#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0; LIB=$R/code/eval_frozen_c3r2/lib_gate_r2.json
cd $R; tmux new-session -d -s pp-lch2svce "source env.sh; cd code/eval; CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P lch2.py serve_multi 36121 e=$R/runs/lch/v2e.pt > $R/scripts/logs/lch2svce.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-lch2svce -F '#{pane_pid}') pp-lch2svce gpu3 LCH v2e service port 36121 running" >> runs/PIDS.txt
until curl -s -X POST localhost:36121/e/score -H 'Content-Type: application/json' -d '{"texts":["Task: x\nRecent steps:\n(none)\nCandidate action: look around"]}' | grep -q tau; do sleep 20; tmux has-session -t pp-lch2svce || { echo SVC_FAIL; tail $R/scripts/logs/lch2svce.log; exit 1; }; done; echo SVC_UP
cd $R/code/eval; LCH_URL=http://127.0.0.1:36121/e PATCHES=$LIB META_SHIELD=learned LOO_DEMO=1 SCIWORLD_URL=http://127.0.0.1:36112 $T run_eval.py --env sciworld --split train_ho2all --method meta_replay --max-steps 30 --workers 3 --run-id r3/v2e_learned_ho --llm-url http://127.0.0.1:8125/v1 2>&1 | grep -E "SUMMARY|Traceback"
echo P01E_DONE
