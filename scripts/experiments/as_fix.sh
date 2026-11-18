#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; S=$R/scripts; L=$R/code/MetaAct/logs/runs; A=$R/code/MetaAct/artifacts/baselines
for e in textcraft:tc alfworld:alf sciworld:sw webshop:ws; do IFS=: read env b <<< "$e"
  [ -d $L/ma_agentspec_${env}_test ] && [ ! -d $L/ma_agentspec_${env}_test_badrules ] && mv $L/ma_agentspec_${env}_test $L/ma_agentspec_${env}_test_badrules
  bash $S/ma_run.sh agentspec $env test 8124 - --agentspec_rules $A/agentspec_${b}_rules.json
done; echo AS_FIX_DONE
