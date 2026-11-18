#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# LCH dev comparison on ScienceWorld (serial Qwen, deterministic): hand-written shield vs none vs learned vs both
# usage: lch_dev.sh TAG HEAD.pt   (LCH service on 36110 must serve HEAD.pt)
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0; cd $PINPATCH_ROOT/code/eval
T=$1; P=$PINPATCH_ENVS/tc/bin/python
run() { META_SHIELD=$1 $P run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id lch_${T}_sci_$1_dev --llm-url http://127.0.0.1:$2/v1 2>&1 | grep -E "SUMMARY|Traceback|Error"; }
run hand 8155 & run none 8156 & wait
run learned 8155 & run both 8156 & wait
echo LCH_DEV_DONE $T
