#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# large held-out (train tasks never used anywhere), 8 shards x serial servers: base vs gate-merged vs random-merged
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
P8=(8123 8133 8124 8134 8125 8135 8126 8136)
for m in base gate random; do
  for s in 0 1 2 3 4 5 6 7; do
    PA=""; [ $m != base ] && PA=$R/runs/repair2/patches_${m}_merged.json
    PATCHES=$PA LOO_DEMO=1 LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split train_ho2s$s --method meta_replay --max-steps 30 --workers 3 --run-id ho2_${m}_s$s --llm-url http://127.0.0.1:${P8[$s]}/v1 2>&1 | grep -E "Traceback" &
  done
done; wait; echo HO2_DONE
