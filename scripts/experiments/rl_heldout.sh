#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# held-out TRAIN evaluation (120 tasks unused by LCH and RL; leave-one-out demo retrieval), serial LoRA servers
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
NAME=$1; MODEL=$2; PORT=$3; SCI=$4
for m in both none; do SCIWORLD_URL=$SCI LLM_MODEL=$MODEL LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=$m PYTHONHASHSEED=0 $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split train_heldout --method meta_replay --max-steps 30 --workers 4 --run-id ho_${NAME}_${m} --llm-url http://127.0.0.1:$PORT/v1 2>&1 | grep -E "SUMMARY|Traceback" & done; wait; echo HELDOUT_DONE $NAME
