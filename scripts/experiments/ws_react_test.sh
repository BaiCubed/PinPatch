#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0
cd $PINPATCH_ROOT/code/eval
$PINPATCH_ENVS/tc/bin/python run_eval.py --env webshop --split test --method react --max-steps 10 --workers 4 --run-id b1_ws_react_test_serial --llm-url http://127.0.0.1:8123/v1 2>&1 | grep -E "SUMMARY|Traceback|Error"
echo "RUN_DONE react webshop"
