#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# step 2 iteration 2: rollouts it1 with the it1 gate adapter -> train it2 gate (init it1 gate, GPU0)
# -> after the it1 dev arms finish, restart the serial LoRA server with rlg2 and run its dev arms
R=$PINPATCH_ROOT; cd $R; source env.sh; LG=runs/logs
until grep -q "ROLLOUT_DONE 1" $LG/rl_roll1.log; do sleep 60; done
cd code/eval; CUDA_VISIBLE_DEVICES=0 $PINPATCH_ENVS/main/bin/python rl_train.py 1 $R/runs/rl/lora_it2_gate --gate --init $R/runs/rl/lora_it1_gate 2>&1 | grep -E "gate:|it1:|saved|Error|Trace"
cd $R; until [ -f $LG/rl_base.done ]; do sleep 60; done
tmux kill-session -t pp-vllmLS4; echo "$(date +%F_%T) - pp-vllmLS4 stopped(restart with rlg2)" >> runs/PIDS.txt; sleep 10
tmux new -d -s pp-vllmLS4 "source env.sh; CUDA_VISIBLE_DEVICES=4 $PINPATCH_ENVS/main/bin/vllm serve $R/models/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8180 --max-model-len 8192 --seed 0 --gpu-memory-utilization 0.3 --max-num-seqs 1 --no-enable-prefix-caching --enable-lora --max-lora-rank 32 --lora-modules rlu=$R/runs/rl/lora_it1_uniform rlg=$R/runs/rl/lora_it1_gate rlg2=$R/runs/rl/lora_it2_gate > runs/vllm/lora_serial_gpu4_b.log 2>&1"
echo "$(date +%F_%T) $(tmux list-panes -t pp-vllmLS4 -F "#{pane_pid}") pp-vllmLS4 vllm-serial-LoRA-gpu4-port8180 (+rlg2) running" >> runs/PIDS.txt
for i in $(seq 1 90); do curl -s -m 3 127.0.0.1:8180/v1/models >/dev/null && break; sleep 5; done
cd code/eval
for m in both none; do LLM_MODEL=rlg2 LCH_TAU=0.9761 META_SHIELD=$m PYTHONHASHSEED=0 $PINPATCH_ENVS/tc/bin/python run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id rl_it2_rlg2_${m}_dev --llm-url http://127.0.0.1:8180/v1 2>&1 | grep -E "SUMMARY|Traceback" & done; wait
echo RL_PIPELINE2_DONE
