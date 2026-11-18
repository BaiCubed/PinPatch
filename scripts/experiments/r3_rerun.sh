#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# R3 reruns (pre-registered PROGRESS.md 2026-11-15). Waits for P13_DONE, then one Qwen serial server on GPU $1 port 8172.
R=$PINPATCH_ROOT; cd $R; source env.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
G=${1:-7}; PORT=8172; S=pp-vllmS8172; L=$R/code/MetaAct/logs/runs; A=$R/code/MetaAct/artifacts/baselines
until grep -q P13_DONE runs/logs/p13.log 2>/dev/null; do sleep 300; done
tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_UTIL=0.2 VLLM_KV=1610612736 bash scripts/vllm.sh $G $PORT serial > runs/vllm/r3_$PORT.log 2>&1"
  echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu$G-port$PORT (R3 reruns) running" >> runs/PIDS.txt; }
until curl -s localhost:$PORT/v1/models | grep -q Qwen; do sleep 20; tmux has-session -t $S 2>/dev/null || { echo SERVER_FAIL; exit 1; }; done; echo SERVER_UP
U=http://127.0.0.1:$PORT/v1
# 1 ExpeL (fixed placement), 4 envs
for e in textcraft alfworld sciworld webshop; do s=$(echo $e | sed 's/textcraft/tc/;s/alfworld/alf/;s/sciworld/sci/;s/webshop/ws/')
  st=30; [ $e = textcraft ] && st=40; [ $e = webshop ] && st=10; d=runs/b1_${s}_expel_test_serial
  [ -d $d ] && [ ! -d ${d}_promptbug ] && mv $d ${d}_promptbug
  ( cd code/eval; $T baselines.py --env $e --split test --method expel --max-steps $st --limit 0 --workers 4 --run-id b1_${s}_expel_test_serial --llm-url $U 2>&1 | grep -E 'SUMMARY|Traceback' ) &
done; wait; echo R3_EXPEL_DONE
# 2 OntoGuard with the authors' published KBs, 4 envs
for e in textcraft alfworld sciworld webshop; do b=$(echo $e | sed 's/textcraft/tc/;s/alfworld/alf/;s/sciworld/sw/;s/webshop/ws/')
  [ -d $L/ma_ontoguard_${e}_test ] && [ ! -d $L/ma_ontoguard_${e}_test_ourkb ] && mv $L/ma_ontoguard_${e}_test $L/ma_ontoguard_${e}_test_ourkb
  bash scripts/ma_run.sh ontoguard $e test $PORT - --ontoguard_kb $A/ontoguard_${b}_kb.json &
done; wait; echo R3_OG_DONE
# 3 ReflAct tolerant parse (appendix), ALFWorld + ScienceWorld
for e in alfworld sciworld; do
  [ -d $L/ma_reflact_${e}_test ] && [ ! -d $L/ma_reflact_${e}_test_strictparse ] && mv $L/ma_reflact_${e}_test $L/ma_reflact_${e}_test_strictparse
  REFLACT_TOLERANT=1 bash scripts/ma_run.sh reflact $e test $PORT - &
done; wait; echo R3_REFLACT_DONE
# keep the appendix run under its own name and restore the main-table run
for e in alfworld sciworld; do mv $L/ma_reflact_${e}_test $L/ma_reflact_${e}_test_tolerant && mv $L/ma_reflact_${e}_test_strictparse $L/ma_reflact_${e}_test; done
echo R3_DONE; tmux kill-session -t $S; echo "$(date +%F_%T) - $S stopped (R3_DONE)" >> runs/PIDS.txt
