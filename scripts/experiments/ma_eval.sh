#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# usage: ma_eval.sh ENV SPLIT RUN_ID PORT [run_episode_eval args...]  -> MetaAct full pipeline on our AgentGym env
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0; export MA_SCRIPT=run_episode_eval
cd $PINPATCH_ROOT/code/eval
ENV=$1; SPLIT=$2; RID=$3; PORT=$4; shift 4
$PINPATCH_ENVS/ma/bin/python metaact_ag.py $ENV $SPLIT $RID $PORT -- "$@" 2>&1 | grep -E "episodes|!!|Error|Trace|success_rate|gear-synth"
echo "MA_RUN_DONE metaact $RID"
