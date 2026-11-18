#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# MetaAct-synth: GEAR synthesis + replay verification from the train evidence (Qwen2.5-7B, 32k server), then test
# usage: ma_synth_test.sh ENV FAMILY(or -) REPLACES TESTPORT [test flags...]
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
ENV=$1; FAM=$2; REP=$3; PORT=$4; shift 4; C=$PINPATCH_ROOT/runs/ma_art/gear_${ENV}_synth.jsonl
F=""; [ "$FAM" != "-" ] && F="--family $FAM"
$PINPATCH_ENVS/ma/bin/python metaact_ag.py raw gear_synth_run $ENV 8128 -- --runs logs/runs/ma_synthsrc_$ENV --dataset $ENV $F --cache $C --synth_run_id ma_gear_$ENV 2>&1 | tail -6
echo "specs: $(wc -l < $C 2>/dev/null)"
bash $PINPATCH_ROOT/scripts/ma_eval.sh $ENV test ma_metaact_${ENV}_test $PORT "$@" --gear_synth $C --gear_synth_replaces $REP --seed 42
echo SYNTH_TEST_DONE $ENV
