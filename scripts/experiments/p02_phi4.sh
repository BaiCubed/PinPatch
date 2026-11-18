#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P0-2: second backbone microsoft/phi-4 (amended from Qwen2.5-14B), TEST once per config (pre-registered in PROGRESS.md round 3)
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python; export PYTHONHASHSEED=0
M14=$R/models/phi-4; PORT=8140
$P -c "from huggingface_hub import snapshot_download as s; print(s('microsoft/phi-4', local_dir='$M14'))" 2>&1 | tail -1
ls $M14/*.safetensors | wc -l
# frozen ALFWorld controller: identical copy except the served model name comes from LLM_MODEL (old copy hard-codes 7B)
mkdir -p $R/code/eval_frozen_alf_m; cp $R/code/eval_frozen_alf/run_eval.py $R/code/eval_frozen_alf_m/
sed -i "s|body = dict(model=MODEL,|body = dict(model=os.environ.get('LLM_MODEL', MODEL),|" $R/code/eval_frozen_alf_m/run_eval.py
diff $R/code/eval_frozen_alf/run_eval.py $R/code/eval_frozen_alf_m/run_eval.py
cd $R; tmux new-session -d -s pp-vllmphi4 "VLLM_MODEL=$M14 VLLM_UTIL=0.38 VLLM_KV=2147483648 bash scripts/vllm.sh 4 $PORT serial > runs/vllm/p02_$PORT.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-vllmphi4 -F '#{pane_pid}') pp-vllmphi4 vllm-serial-gpu4-port$PORT (P0-2 phi-4) running" >> runs/PIDS.txt
until curl -s localhost:$PORT/v1/models | grep -q phi-4; do sleep 20; tmux has-session -t pp-vllmphi4 2>/dev/null || { echo SERVER_FAIL; tail -20 $R/runs/vllm/p02_$PORT.log; exit 1; }; done; echo SERVER_UP
export LLM_MODEL=$M14; U=http://127.0.0.1:$PORT/v1
( cd $R/code/eval_frozen_c3r2; PATCHES=$R/code/eval_frozen_c3r2/lib_gate_r2.json LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id p02_phi4/sci_ours_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval_frozen_alf_m; $T run_eval.py --env alfworld --split test --method meta_alf --max-steps 30 --workers 4 --run-id p02_phi4/alf_ours_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval; $T run_eval.py --env sciworld --split test --method react --max-steps 30 --workers 4 --run-id p02_phi4/sci_react_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval; $T run_eval.py --env alfworld --split test --method react --max-steps 30 --workers 4 --run-id p02_phi4/alf_react_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval; $T baselines.py --env sciworld --split test --method reflexion --max-steps 30 --limit 0 --workers 4 --run-id p02_phi4/sci_reflexion_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval; $T baselines.py --env alfworld --split test --method reflexion --max-steps 30 --limit 0 --workers 4 --run-id p02_phi4/alf_reflexion_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
wait; echo P02_DONE
