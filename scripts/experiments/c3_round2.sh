#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# evolution round 2 for one method (gate | random): base with the round-1 library on the 1303 earlier train tasks ->
# orders -> this method's B=2 edits -> evolve on top of the library -> held-out 293 with the round-2 library
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0; M=$1; export C3_DIR=$R/runs/repair3_$M; LIB=$R/runs/repair2/patches_${M}_merged.json
PATCHES=$LIB $T c3_scale.py base_ids $R/runs/repair/base.jsonl $R/runs/repair2/base.jsonl 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=4 PATCHES=$LIB $P c3_scale.py orders 2>&1 | grep -E "ORDERS_DONE|Error|Trace"
PATCHES=$LIB C3_METHODS=$M $T c3_scale.py search 2 2>&1 | grep -E "edit runs|SEARCH_DONE|Error|Trace"
PATCHES=$LIB PATCHES_BASE=$LIB $T evolve.py $M 2 $C3_DIR/patches_${M}_r2.json 2>&1 | grep -E "B=|kept|library|Error|Trace"
P8=(8123 8133 8124 8134 8125 8135 8126 8136)
for s in 0 1 2 3 4 5 6 7; do
  PATCHES=$C3_DIR/patches_${M}_r2_lib.json LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_ho2s$s --method meta_replay --max-steps 30 --workers 2 --run-id ho2_${M}r2_s$s --llm-url http://127.0.0.1:${P8[$s]}/v1 2>&1 | grep Traceback &
done; wait
echo ROUND2_DONE $M
