#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; export PYTHONHASHSEED=0
for r in c3r1 c4 c3r2; do $PINPATCH_ENVS/tc/bin/python flip_audit.py accepted $r 2>&1 | grep -E "^c3|^c4|Error|Trace"; done; echo FLIP_DONE
