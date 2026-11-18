#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# reruns after the port fixes (step-costed teleport macro; ALFWorld with grounding on)
S=$PINPATCH_ROOT/scripts; E=$1
case $E in
 sciworld) bash $S/ma_eval.sh sciworld train30 ma_synthsrc_sciworld 8160 --variant full --no_gear_focus --seed 42
           bash $S/ma_synth_test.sh sciworld - focus 8160 --variant full ;;
 alfworld) bash $S/ma_eval.sh alfworld train140 ma_synthsrc_alfworld 8161 --variant static --no_gear_repeat --seed 42
           bash $S/ma_synth_test.sh alfworld alfworld repeat 8161 --variant static ;;
 rpms_sci) bash $S/ma_rpms.sh sciworld 30 8162 ;;
esac
echo RERUN_DONE $E
