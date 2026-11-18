#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# plan B step 2 iteration 1 on ScienceWorld, end to end on the server:
# rollouts it0 (base, K=4, TRAIN) -> train RL-uniform and RL-gate adapters (GPU4) -> serial LoRA server (GPU4)
# -> dev arms with the learned+hand guard (both, tau 0.9761) and with no guard (internalization)
R=$PINPATCH_ROOT; cd $R; source env.sh; LG=runs/logs
until grep -q "ROLLOUT_DONE 0" $LG/rl_roll0.log; do sleep 60; done
cd code/eval; export CUDA_VISIBLE_DEVICES=4; P=$PINPATCH_ENVS/main/bin/python
$P rl_train.py 0 $R/runs/rl/lora_it1_uniform 2>&1 | grep -v "Loading\|it/s" | tail -4
$P rl_train.py 0 $R/runs/rl/lora_it1_gate --gate 2>&1 | grep -v "Loading\|it/s" | tail -5
unset CUDA_VISIBLE_DEVICES; cd $R
tmux new -d -s pp-vllmLS4 "source env.sh; CUDA_VISIBLE_DEVICES=4 $PINPATCH_ENVS/main/bin/vllm serve $R/models/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8180 --max-model-len 8192 --seed 0 --gpu-memory-utilization 0.3 --max-num-seqs 1 --no-enable-prefix-caching --enable-lora --max-lora-rank 32 --lora-modules rlu=$R/runs/rl/lora_it1_uniform rlg=$R/runs/rl/lora_it1_gate > runs/vllm/lora_serial_gpu4.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-vllmLS4 -F "#{pane_pid}") pp-vllmLS4 vllm-serial-LoRA-gpu4-port8180 (step2 dev eval) running" >> runs/PIDS.txt
for i in $(seq 1 90); do curl -s -m 3 127.0.0.1:8180/v1/models >/dev/null && break; sleep 5; done
cd code/eval
dev() { LLM_MODEL=$1 LCH_TAU=0.9761 META_SHIELD=$2 PYTHONHASHSEED=0 $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id rl_it1_$1_$2_dev --llm-url http://127.0.0.1:8180/v1 2>&1 | grep -E "SUMMARY|Traceback"; }
dev rlu both & dev rlg both & wait
dev rlu none & dev rlg none & wait
echo RL_PIPELINE_DONE
