#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# usage: ma_run.sh METHOD ENV SPLIT PORT SHARD(i/n or -) [extra args...]  -> MetaAct-repo baseline on our AgentGym env
source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0
cd $PINPATCH_ROOT/code/eval
M=$1; ENV=$2; SPLIT=$3; PORT=$4; SH=$5; shift 5
X=""; [ "$SH" != "-" ] && X="--shard $SH"
$PINPATCH_ENVS/ma/bin/python metaact_ag.py $ENV $SPLIT ma_${M}_${ENV}_${SPLIT} $PORT -- --baseline $M $X "$@" 2>&1 | grep -E "episodes|!!|Error|Trace|success_rate"
echo "MA_RUN_DONE $M $ENV $SPLIT $SH"
