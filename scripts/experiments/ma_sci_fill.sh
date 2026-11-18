#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval; C=$PINPATCH_ROOT/runs/ma_art/gear_sciworld_synth.jsonl
$PINPATCH_ENVS/ma/bin/python metaact_ag.py raw gear_synth_run sciworld 8128 -- --runs logs/runs/ma_synthsrc_sciworld --dataset sciworld --cache $C --synth_run_id ma_gear_sciworld_fill --skip_done $C 2>&1 | tail -4
echo "specs: $(wc -l < $C)"
bash $PINPATCH_ROOT/scripts/ma_eval.sh sciworld test ma_metaact_sciworld_test 8160 --variant full --gear_synth $C --gear_synth_replaces focus --seed 42
echo RERUN_DONE sciworld
