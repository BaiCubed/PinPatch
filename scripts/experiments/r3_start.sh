#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# round 3 start: downloads (14B, AgentTraj-L) + held-out references + LCH-v2 data collection
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python
export PYTHONHASHSEED=0; LIB=$R/code/eval_frozen_c3r2/lib_gate_r2.json
( $P -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-14B-Instruct', local_dir='$R/models/Qwen2.5-14B-Instruct')); print(s('AgentGym/AgentTraj-L', repo_type='dataset', local_dir='$R/data/AgentTraj-L', allow_patterns=['alfworld_train.json','sciworld_train.json','textcraft_train.json','webshop_train.json']))" 2>&1 | grep -vi warn | tail -3; echo DL_DONE ) &
P4=(8123 8124 8125 8126); SW=(http://127.0.0.1:36102 http://127.0.0.1:36112); k=0
run() { local name=$1 split=$2; shift 2
  env "$@" SCIWORLD_URL=${SW[$((k%2))]} $T run_eval.py --env sciworld --split $split --method meta_replay --max-steps 30 --workers 3 --run-id r3/$name --llm-url http://127.0.0.1:${P4[$((k%4))]}/v1 2>&1 | grep -E "SUMMARY|Traceback" | sed "s|^|$name |" &
  k=$((k+1)); }
for m in hand learned none; do run ref_${m}_ho train_ho2all PATCHES=$LIB META_SHIELD=$m LCH_TAU=0.9761 LOO_DEMO=1; done
run lch2a_noshield train_lch2a PATCHES=$LIB META_SHIELD=none LOO_DEMO=1
run lch2b_noshield_t07 train_lch2b PATCHES=$LIB META_SHIELD=none LOO_DEMO=1 ASK_TEMP=0.7
wait; echo R3START_DONE
