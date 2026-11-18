#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# FOCUS_POLICY variant (the policy picks focus targets): base dev arms, rollouts, gate training, dev arms with adapter
R=$PINPATCH_ROOT; cd $R; source env.sh; LG=runs/logs; M=$R/models/Qwen2.5-7B-Instruct
export FOCUS_POLICY=1
( cd code/eval; for m in both none; do LCH_TAU=0.9761 META_SHIELD=$m PYTHONHASHSEED=0 $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id fp_base_${m}_dev --llm-url http://127.0.0.1:$([ $m = both ] && echo 8155 || echo 8156)/v1 2>&1 | grep -E "SUMMARY|Traceback" & done; wait ) &
RLP=fp bash scripts/rl_rollout2.sh 0 $M 4
cd code/eval; RLP=fp CUDA_VISIBLE_DEVICES=3 $PINPATCH_ENVS/main/bin/python rl_train.py 0 $R/runs/rl/lora_fp1_gate --gate 2>&1 | grep -E "fp0:|it0:|saved|Error|Trace"
cd $R; tmux new -d -s pp-vllmLS3 "source env.sh; CUDA_VISIBLE_DEVICES=3 $PINPATCH_ENVS/main/bin/vllm serve $M --host 127.0.0.1 --port 8181 --max-model-len 8192 --seed 0 --gpu-memory-utilization 0.3 --max-num-seqs 1 --no-enable-prefix-caching --enable-lora --max-lora-rank 32 --lora-modules fpg=$R/runs/rl/lora_fp1_gate > runs/vllm/lora_serial_gpu3.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-vllmLS3 -F "#{pane_pid}") pp-vllmLS3 vllm-serial-LoRA-gpu3-port8181 (fp eval) running" >> runs/PIDS.txt
for i in $(seq 1 90); do curl -s -m 3 127.0.0.1:8181/v1/models >/dev/null && break; sleep 5; done
cd code/eval; for m in both none; do LLM_MODEL=fpg LCH_TAU=0.9761 META_SHIELD=$m PYTHONHASHSEED=0 $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id fp_rlg_${m}_dev --llm-url http://127.0.0.1:8181/v1 2>&1 | grep -E "SUMMARY|Traceback" & done; wait
wait; echo RL_FP_DONE
