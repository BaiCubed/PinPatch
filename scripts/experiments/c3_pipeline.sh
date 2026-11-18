#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# C3: orders -> evolve (gate/random/recency at B=2, all) -> dev (both + LCH strict) with each patch set
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; P=$PINPATCH_ENVS/main/bin/python; T=$PINPATCH_ENVS/tc/bin/python
CUDA_VISIBLE_DEVICES=4 $P repair_eval.py ../../runs/repair/base.jsonl ../../runs/repair/edits.jsonl 2>&1 | grep -E "gate vs|mean tries" | tail -3
for mb in gate:2 random:2 recency:2 all:0; do m=${mb%:*}; b=${mb#*:}; PYTHONHASHSEED=0 $T evolve.py $m $b ../../runs/repair/patches_${m}_B$b.json 2>&1 | grep -vE "^\s*$" | tail -6; done
echo EVOLVE_DONE
P8=(8123 8133 8124 8134)
k=0; for mb in gate:2 random:2 recency:2 all:0; do m=${mb%:*}; b=${mb#*:}; PATCHES=$R/runs/repair/patches_${m}_B$b.json LCH_TAU=0.9761 META_SHIELD=both PYTHONHASHSEED=0 $T run_eval.py --env sciworld --split dev --method meta_replay --max-steps 30 --workers 4 --run-id c3_${m}_B${b}_dev --llm-url http://127.0.0.1:${P8[$k]}/v1 2>&1 | grep -E "SUMMARY|Traceback" & k=$((k+1)); done; wait
echo C3_DONE
