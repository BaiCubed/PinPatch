#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# plan B step 2 rollouts on TRAIN: K sampled episodes per task (temperature 1 at LLM decision points only),
# learned constraint head as the guard; every LLM decision logged. usage: rl_rollout.sh IT MODEL_NAME K
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
IT=$1; MN=$2; K=$3
for k in $(seq 0 $((K-1))); do
  U=http://127.0.0.1:36112; [ $((k%2)) = 1 ] && U=http://127.0.0.1:36102
  SCIWORLD_URL=$U LLM_MODEL=$MN ASK_TEMP=1.0 LOG_ASKS=1 LOO_DEMO=1 META_SHIELD=learned LCH_TAU=0.9761 PYTHONHASHSEED=0 \
    $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split train_rl --method meta_replay --max-steps 30 \
    --workers 16 --seed $k --run-id rl/it${IT}_s$k --llm-url http://127.0.0.1:8170/v1 2>&1 | grep -E "SUMMARY|Traceback" &
done; wait; echo ROLLOUT_DONE $IT
