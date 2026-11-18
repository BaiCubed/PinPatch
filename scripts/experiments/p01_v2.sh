#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P0-1: build v2 data (GPU5), credit labels, train variants a-d, serve all heads, held-out 293 per variant (v2-only, full system)
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $R/runs/lch/v2.npz ] || CUDA_VISIBLE_DEVICES=5 $P lch2.py build $R/runs/r3/lch2a_noshield $R/runs/r3/lch2b_noshield_t07 2>&1 | grep -E "episodes|features|v2 data|Error|Trace"
[ -f $R/runs/lch/v2c.npy ] || $P lch2.py credit 2>&1 | grep -E "credit|Error|Trace"
for v in a b c d; do [ -f $R/runs/lch/v2$v.pt ] || $P lch2.py train $v $R/runs/lch/v2$v.pt 2>&1 | grep -E "^$v |Error|Trace"; done
cd $R; tmux new-session -d -s pp-lch2svc "source env.sh; cd code/eval; CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P lch2.py serve_multi 36120 a=$R/runs/lch/v2a.pt b=$R/runs/lch/v2b.pt c=$R/runs/lch/v2c.pt d=$R/runs/lch/v2d.pt > $R/scripts/logs/lch2svc.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-lch2svc -F '#{pane_pid}') pp-lch2svc gpu5 LCH v2 multi-head service port 36120 running" >> runs/PIDS.txt
until curl -s -X POST localhost:36120/a/score -H 'Content-Type: application/json' -d '{"texts":["Task: x\nRecent steps:\n(none)\nCandidate action: look around"]}' | grep -q tau; do sleep 20; tmux has-session -t pp-lch2svc || { echo SVC_FAIL; tail $R/scripts/logs/lch2svc.log; exit 1; }; done; echo SVC_UP
cd $R/code/eval; LIB=$R/code/eval_frozen_c3r2/lib_gate_r2.json; P4=(8123 8125 8126 8123); SW=(http://127.0.0.1:36102 http://127.0.0.1:36112); k=0
for v in a b c d; do
  LCH_URL=http://127.0.0.1:36120/$v PATCHES=$LIB META_SHIELD=learned LOO_DEMO=1 SCIWORLD_URL=${SW[$((k%2))]} $T run_eval.py --env sciworld --split train_ho2all --method meta_replay --max-steps 30 --workers 3 --run-id r3/v2${v}_learned_ho --llm-url http://127.0.0.1:${P4[$k]}/v1 2>&1 | grep -E "SUMMARY|Traceback" | sed "s|^|v2$v |" &
  k=$((k+1))
done; wait; echo P01_HO_DONE
