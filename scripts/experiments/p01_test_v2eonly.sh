#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0; C=$R/code/eval_frozen_lchv2; cd $C
sha256sum -c $R/runs/frozen_lchv2/SHA256SUMS
LCH_URL=http://127.0.0.1:36121/e META_SHIELD=learned PATCHES=$C/lib_gate_r2.json $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id frozen_lchv2/v2e_only_test --llm-url http://127.0.0.1:8123/v1 2>&1 | grep -E "SUMMARY|Traceback"
echo V2EONLY_TEST_DONE
