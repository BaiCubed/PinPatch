#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# usage: run_one.sh PORT ENV STEPS SHORT METHOD  (one env, test split; same run-id scheme as run_baselines.sh)
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0
cd $PINPATCH_ROOT/code/eval
$PINPATCH_ENVS/tc/bin/python baselines.py --env $2 --split test --method $5 --max-steps $3 --run-id b1_$4_$5_test_serial --workers 4 --llm-url http://127.0.0.1:$1/v1 2>&1 | grep -E "SUMMARY|Traceback|Error"
echo "RUN_DONE $5 $2 (run_one)"
