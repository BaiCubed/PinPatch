#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# usage: run_baselines.sh PORT SPLIT LIMIT METHOD...   runs each method on textcraft, alfworld, sciworld (serial vLLM at PORT)
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0
cd $PINPATCH_ROOT/code/eval
PORT=$1; SPLIT=$2; LIM=$3; shift 3
for m in "$@"; do for e in textcraft:40:tc alfworld:30:alf sciworld:30:sci; do
  IFS=: read env st short <<< "$e"
  tag=${TAG-$([ "$LIM" = 0 ] && echo "" || echo "_smoke")}
  $PINPATCH_ENVS/tc/bin/python baselines.py --env $env --split $SPLIT --method $m --max-steps $st --limit $LIM \
    --run-id b1_${short}_${m}_${SPLIT}${tag}_serial --workers 4 --llm-url http://127.0.0.1:$PORT/v1 2>&1 | grep -E "SUMMARY|Traceback|Error" 
  echo "RUN_DONE $m $env $($PINPATCH_ENVS/tc/bin/python -c "import json;s=json.load(open('../../runs/b1_${short}_${m}_${SPLIT}${tag}_serial/summary.json'));print(s['success'],'/',s['n'])" 2>&1)"
done; done
echo BASELINES_DONE "$@"
