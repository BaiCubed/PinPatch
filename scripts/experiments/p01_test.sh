#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P0-1 frozen TEST: full system (gate r2 library) with v2e-only and hand+v2e; one run each
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
F=$R/runs/frozen_lchv2; mkdir -p $F; C=$R/code/eval_frozen_lchv2; mkdir -p $C
cp $R/code/eval/run_eval.py $R/code/eval/lch.py $R/code/eval/lch2.py $C/; cp $R/runs/lch/v2e.pt $C/; cp $R/code/eval_frozen_c3r2/lib_gate_r2.json $C/
cd $C; sha256sum run_eval.py lch.py lch2.py v2e.pt lib_gate_r2.json > $F/SHA256SUMS
cat > $F/FREEZE.json <<J
{"frozen_at": "$(date '+%F %T')", "env": "sciworld (AgentGym official test, 200)",
 "method": "PinPatch full system (gate r2 library) with learned constraint v2e (label-cleaned, layer 20, strict tau)",
 "configs": {"v2e_only_test": "META_SHIELD=learned, LCH_URL=v2e service", "hand_v2e_test": "META_SHIELD=both, LCH_URL=v2e service"},
 "code_commit_eval": "$(git -C $R/code/eval log -1 --format=%H 2>/dev/null || echo see SHA256SUMS)",
 "selection": "held-out 293 only (pre-registered): v1 127, v2a 129, v2e 130 (neg-score 57), hand 135; v2b/c/d never block",
 "step_cap": 30, "seed": 0, "PYTHONHASHSEED": 0}
J
U=http://127.0.0.1:36121/e
LCH_URL=$U META_SHIELD=learned PATCHES=$C/lib_gate_r2.json $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id frozen_lchv2/v2e_only_test --llm-url http://127.0.0.1:8123/v1 2>&1 | grep -E "SUMMARY|Traceback" &
LCH_URL=$U META_SHIELD=both PATCHES=$C/lib_gate_r2.json $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id frozen_lchv2/hand_v2e_test --llm-url http://127.0.0.1:8125/v1 2>&1 | grep -E "SUMMARY|Traceback" &
wait; echo LCHV2_TEST_DONE
