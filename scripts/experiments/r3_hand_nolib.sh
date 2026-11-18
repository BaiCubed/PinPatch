#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; export PYTHONHASHSEED=0
git -C $R/code/eval log --oneline -1 2>/dev/null; md5sum run_eval.py
META_SHIELD=hand $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id r3/test_hand_nolib --llm-url http://127.0.0.1:8125/v1 2>&1 | grep -E "SUMMARY|Traceback"
echo HANDNOLIB_DONE
