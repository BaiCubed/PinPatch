#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval; A=$PINPATCH_ROOT/runs/ma_art
$PINPATCH_ENVS/ma/bin/python og_demos_ag.py textcraft 60 $A/ontoguard_tc_demos.jsonl 2>&1 | tail -3
ALFWORLD_DATA=$PINPATCH_ROOT/data/alfworld $PINPATCH_ENVS/alf/bin/python og_demos_alf.py $PINPATCH_ROOT/runs/ma_tasks/alfworld_train140.jsonl $A/ontoguard_alf_demos.jsonl 2>&1 | tail -2
echo OG_DEMOS2_DONE
