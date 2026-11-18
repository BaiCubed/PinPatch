#!/bin/bash
# Table 1 (PinPatch rows): the frozen configurations, TEST once each. Needs: env servers (scripts/env_servers.sh) and a
# serial Qwen2.5-7B server on port 8123 (scripts/serve_llm.sh 0 8123) for ScienceWorld/ALFWorld; WebShop and TextCraft
# make no LLM calls. ScienceWorld also needs the constraint-head service (started here on port 36110, GPU 1).
source "$(dirname "$0")/../config.sh"; R=$PINPATCH_ROOT; T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python
U=http://127.0.0.1:${LLM_PORT:-8123}/v1
# TextCraft (100): frozen model, no LLM
( cd $R/code/eval && $T ag_run.py $R/runs/phase2_ag_base/merged eval test mem_analog_gated 0 | tail -2 )
# WebShop (200): frozen controller, no LLM, 10 steps
( cd $R/code/eval_frozen_ws && $T run_eval.py --env webshop --split test --method meta_ws --max-steps 10 --workers 4 --run-id main/ws_ours_test | grep SUMMARY )
# ALFWorld (200): frozen controller, Qwen only when stuck, 30 steps
( cd $R/code/eval_frozen_alf && $T run_eval.py --env alfworld --split test --method meta_alf --max-steps 30 --workers 4 --run-id main/alf_ours_test --llm-url $U | grep SUMMARY )
# ScienceWorld (200): procedure library after two rounds of gate-ordered evolution + hand guard + learned constraint (v1)
tmux has-session -t pp-lch 2>/dev/null || tmux new -d -s pp-lch "cd $R/code/eval; CUDA_VISIBLE_DEVICES=${LCH_GPU:-1} $P lch.py serve $R/runs/lch/v1.pt 36110 > $R/runs/logs/lch_v1.log 2>&1"
until curl -s -XPOST localhost:36110/score -H 'content-type: application/json' -d '{"texts":["x"]}' | grep -q tau; do sleep 10; done
( cd $R/code/eval_frozen_c3r2 && PATCHES=$R/code/eval_frozen_c3r2/lib_gate_r2.json LCH_TAU=0.9761 META_SHIELD=both \
  $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id main/sci_ours_test --llm-url $U | grep SUMMARY )
