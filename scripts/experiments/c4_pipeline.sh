#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# C4: orders (GPU5) -> budgeted search B=2 (gate, random, simple) -> evolve each -> held-out 293 for each library
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0 C4_DIR=$R/runs/repair4 REPAIR_PORTS=8123,8133,8124,8134,8125,8135,8126,8136
mkdir -p $C4_DIR
[ -f $C4_DIR/orders.json ] || CUDA_VISIBLE_DEVICES=5 $P evolve_x.py orders 2>&1 | grep -E "ORDERS_DONE|Error|Trace"
$T evolve_x.py search 2 2>&1 | grep -E "edit runs|SEARCH_DONE|Error|Trace|^[0-9]+$"
for m in gate random simple; do $T evolve_x.py evolve $m 2 $C4_DIR/lib_${m}.json 2>&1 | grep -E "B=|kept|rejected|Error|Trace"; done
echo EVOLVE4_DONE
P8=(8123 8133 8124 8134 8125 8135 8126 8136)
for m in gate random simple; do for s in 0 1 2 3 4 5 6 7; do
  PATCHES=$C4_DIR/lib_${m}.json LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_ho2s$s --method meta_replay --max-steps 30 --workers 2 --run-id ho4_${m}_s$s --llm-url http://127.0.0.1:${P8[$s]}/v1 2>&1 | grep Traceback &
done; done; wait
echo C4_DONE
