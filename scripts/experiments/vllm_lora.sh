#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# batched vLLM with runtime LoRA loading, for plan B step 2 rollouts on TRAIN (not used for dev/test numbers)
source $PINPATCH_ROOT/scripts/config.sh; export CUDA_VISIBLE_DEVICES=$1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PINPATCH_ENVS/main/bin/vllm serve $PINPATCH_ROOT/models/Qwen2.5-7B-Instruct --host 127.0.0.1 --port $2 \
  --max-model-len 8192 --seed 0 --gpu-memory-utilization ${3:-0.45} --enable-lora --max-lora-rank 32 --max-loras 4 2>&1
