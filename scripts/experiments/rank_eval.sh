#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
PA=$R/runs/repair2/patches_gate_merged.json; P8=(8123 8133 8124 8134 8125 8135 8126 8136)
for s in 0 1 2 3 4 5 6 7; do LCH_RANK=1 PATCHES=$PA LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_ho2s$s --method meta_replay --max-steps 30 --workers 3 --run-id ho2_gaterank_s$s --llm-url http://127.0.0.1:${P8[$s]}/v1 2>&1 | grep Traceback & done
PATCHES=$PA LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 3 --run-id rk_gate_dev --llm-url http://127.0.0.1:8125/v1 2>&1 | grep -E "SUMMARY|Traceback" &
LCH_RANK=1 PATCHES=$PA LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 3 --run-id rk_gaterank_dev --llm-url http://127.0.0.1:8126/v1 2>&1 | grep -E "SUMMARY|Traceback" &
wait; echo RANK_DONE
