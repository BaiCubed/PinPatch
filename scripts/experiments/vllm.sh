#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# usage: vllm.sh GPU PORT [serial]   serial = deterministic (no batching, no prefix cache), used for all test runs
source $PINPATCH_ROOT/scripts/config.sh
export CUDA_VISIBLE_DEVICES=$1
U=${VLLM_UTIL:-0.3}  # shared cards: fixed share (0.3 ~29GB, 0.2 ~19.6GB is enough for max-len 8192 serial)
X="--gpu-memory-utilization $U"
# VLLM_KV=bytes pins the KV pool (serial needs <0.5GiB at 8192 tokens); expandable segments stop allocator growth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "${VLLM_KV:-}" ] && X="$X --kv-cache-memory $VLLM_KV"
[ "${3:-}" = serial ] && X="$X --max-num-seqs 1 --no-enable-prefix-caching"
X="$X ${VLLM_EXTRA:-}"  # e.g. --enable-lora --lora-modules sft=PATH --max-lora-rank 16
exec $PINPATCH_ENVS/main/bin/vllm serve ${VLLM_MODEL:-$PINPATCH_ROOT/models/Qwen2.5-7B-Instruct} --host 127.0.0.1 --port $2 --max-model-len ${VLLM_MAXLEN:-8192} --seed 0 $X 2>&1
