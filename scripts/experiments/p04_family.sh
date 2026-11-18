#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P0-2b cross-family generalisation (pre-registered in PROGRESS.md 2026-11-15). usage: p04_family.sh NAME HF_ID GPU PORT
# Identical configs to p02_phi4.sh; TEST once per config; resumable (run_eval skips finished ids).
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python; export PYTHONHASHSEED=0
N=$1; HF=$2; G=$3; PORT=$4; M=$R/models/$N; S=pp-vllm$N; cd $R
[ -f $M/config.json ] || $P -c "from huggingface_hub import snapshot_download as s; print(s('$HF', local_dir='$M'))" 2>&1 | tail -1
ls $M/*.safetensors | wc -l
tmux has-session -t $S 2>/dev/null || { tmux new-session -d -s $S "VLLM_MODEL=$M VLLM_UTIL=0.38 VLLM_KV=${P04_KV:-2147483648} VLLM_EXTRA=\"${P04_EXTRA:-}\" bash scripts/vllm.sh $G $PORT serial > runs/vllm/p04_$N.log 2>&1"
  echo "$(date +%F_%T) $(tmux list-panes -t $S -F '#{pane_pid}') $S vllm-serial-gpu$G-port$PORT (P0-2b $N) running" >> runs/PIDS.txt; }
until curl -s localhost:$PORT/v1/models | grep -q "$M"; do sleep 20; tmux has-session -t $S 2>/dev/null || { echo SERVER_FAIL; tail -30 runs/vllm/p04_$N.log; exit 1; }; done; echo SERVER_UP
export LLM_MODEL=$M; U=http://127.0.0.1:$PORT/v1; O=p04_family/$N
( cd $R/code/eval_frozen_c3r2; PATCHES=$R/code/eval_frozen_c3r2/lib_gate_r2.json LCH_TAU=0.9761 META_SHIELD=both $T run_eval.py --env sciworld --split test --method meta_replay --max-steps 30 --workers 4 --run-id $O/sci_ours_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
( cd $R/code/eval_frozen_alf_m; $T run_eval.py --env alfworld --split test --method meta_alf --max-steps 30 --workers 4 --run-id $O/alf_ours_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
for e in sciworld alfworld; do s=${e:0:3}
  ( cd $R/code/eval; $T run_eval.py --env $e --split test --method react --max-steps 30 --workers 4 --run-id $O/${s}_react_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
  ( cd $R/code/eval; $T baselines.py --env $e --split test --method reflexion --max-steps 30 --limit 0 --workers 4 --run-id $O/${s}_reflexion_test --llm-url $U 2>&1 | grep -E "SUMMARY|Traceback" ) &
done; wait; echo P04_DONE $N
tmux kill-session -t $S; echo "$(date +%F_%T) - $S stopped (P04_DONE $N)" >> runs/PIDS.txt
