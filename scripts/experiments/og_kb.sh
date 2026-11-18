#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# OntoGuard: parse demos with Qwen2.5-7B, then build the ontology (repo script, our artifact dir)
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
export MA_ART=$PINPATCH_ROOT/runs/ma_art; B=$1; ENV=$2; PORT=$3
$PINPATCH_ENVS/ma/bin/python metaact_ag.py raw ontoguard_build $ENV $PORT -- --bench $B --stage parse 2>&1 | tail -3
$PINPATCH_ENVS/ma/bin/python metaact_ag.py raw ontoguard_build $ENV $PORT -- --bench $B --stage kb 2>&1 | tail -3
echo OG_KB_DONE $B
