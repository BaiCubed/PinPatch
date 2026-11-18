#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# C3 at scale: base (~1600 new train tasks) -> orders (GPU4) -> budgeted search B=2 -> evolve gate/random/recency
# -> dev and held-out (120) evaluation of each patch set and of the unpatched base, all both + LCH strict
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
export PYTHONHASHSEED=0
$T c3_scale.py base 1600 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=4 $P c3_scale.py orders 2>&1 | grep -E "ORDERS_DONE|Error|Trace"
$T c3_scale.py search 2 2>&1 | grep -E "edit runs|SEARCH_DONE|Error|Trace"
for m in gate random recency; do C3_DIR=$R/runs/repair2 $T evolve.py $m 2 $R/runs/repair2/patches_${m}_B2.json 2>&1 | grep -E "B=|kept"; done
echo EVOLVE2_DONE
P8=(8123 8133 8124 8134 8125 8135 8126 8136); k=0
for m in gate random recency; do
  PATCHES=$R/runs/repair2/patches_${m}_B2.json LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id c3s_${m}_dev --llm-url http://127.0.0.1:${P8[$k]}/v1 2>&1 | grep -E "SUMMARY|Traceback" & k=$((k+1))
  PATCHES=$R/runs/repair2/patches_${m}_B2.json LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_heldout --method meta_replay --max-steps 30 --workers 4 --run-id c3s_${m}_ho --llm-url http://127.0.0.1:${P8[$k]}/v1 2>&1 | grep -E "SUMMARY|Traceback" & k=$((k+1))
done
LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_heldout --method meta_replay --max-steps 30 --workers 4 --run-id c3s_base_ho --llm-url http://127.0.0.1:${P8[$k]}/v1 2>&1 | grep -E "SUMMARY|Traceback" &
wait; echo C3S_DONE
