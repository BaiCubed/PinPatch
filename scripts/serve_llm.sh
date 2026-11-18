#!/bin/bash
# usage: serve_llm.sh GPU PORT [MODEL_DIR]   one deterministic (serial) vLLM server, as used for every test run in the paper:
# --max-num-seqs 1, no prefix caching, max length 8192, seed 0. Default model: models/Qwen2.5-7B-Instruct.
source "$(dirname "$0")/config.sh"
VLLM_MODEL=${3:-$PINPATCH_ROOT/models/Qwen2.5-7B-Instruct} VLLM_UTIL=${VLLM_UTIL:-0.45} VLLM_KV=${VLLM_KV:-2147483648} \
  exec bash $PINPATCH_ROOT/scripts/experiments/vllm.sh $1 $2 serial
