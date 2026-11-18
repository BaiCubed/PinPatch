#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# round 2, evolve + held-out only (the first attempt lost two servers to the memory guard mid-validation)
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0 REPAIR_PORTS=8123,8124,8125,8126; M=$1; export C3_DIR=$R/runs/repair3_$M; LIB=$R/runs/repair2/patches_${M}_merged.json
PATCHES=$LIB PATCHES_BASE=$LIB $T evolve.py $M 2 $C3_DIR/patches_${M}_r2.json 2>&1 | grep -E "B=|kept|library|rejected|Error|Trace"
[ -f $C3_DIR/patches_${M}_r2_lib.json ] || { echo "NO LIBRARY $M"; exit 1; }
P4=(8123 8124 8125 8126)
for s in 0 1 2 3 4 5 6 7; do
  PATCHES=$C3_DIR/patches_${M}_r2_lib.json LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_ho2s$s --method meta_replay --max-steps 30 --workers 2 --run-id ho2_${M}r2_s$s --llm-url http://127.0.0.1:${P4[$((s%4))]}/v1 2>&1 | grep Traceback &
done; wait
echo ROUND2B_DONE $M
