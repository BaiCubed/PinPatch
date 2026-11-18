#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# RPMS: learn episodic memory on a TRAIN slice (--rpms_learn), then evaluate on test with that memory (read-only)
ENV=$1; N=$2; PORT=$3; M=$PINPATCH_ROOT/runs/ma_mem/rpms_$ENV.jsonl
bash $PINPATCH_ROOT/scripts/ma_run.sh rpms $ENV train$N $PORT - --rpms_memory $M --rpms_learn
echo "memory entries: $(wc -l < $M 2>/dev/null)"
bash $PINPATCH_ROOT/scripts/ma_run.sh rpms $ENV test $PORT - --rpms_memory $M
echo RPMS_CHAIN_DONE $ENV
