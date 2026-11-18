# Source this file first:  source scripts/config.sh
# PINPATCH_ROOT  = this repository; outputs go to $PINPATCH_ROOT/runs, models to $PINPATCH_ROOT/models
# PINPATCH_ENVS  = where scripts/setup.sh creates the Python environments (main, tc, sci, alf, ws, ma)
export PINPATCH_ROOT=${PINPATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PINPATCH_ENVS=${PINPATCH_ENVS:-$PINPATCH_ROOT/envs}
export PYTHONHASHSEED=0                    # every run in the paper fixes the hash seed
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $PINPATCH_ROOT/runs/logs $PINPATCH_ROOT/runs/vllm
