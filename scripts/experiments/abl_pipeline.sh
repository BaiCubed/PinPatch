#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# ablations (TEST, one run each) + hyperparameters (held-out 293) + C4 budget B=4; see PROGRESS.md pre-registration
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python
export PYTHONHASHSEED=0
out=$($T $R/scripts/abl_fidelity.py 2>&1 | grep -E "FIDELITY|Error|Trace"); echo "$out"
[[ "$out" == *"FIDELITY 24 / 24"* ]] || { echo FIDELITY_FAIL; exit 1; }
bash $R/scripts/abl_noval.sh 2>&1 | grep -E "^repair|NOVAL|Error|Trace"
FULL=$R/code/eval_frozen_c3r2/lib_gate_r2.json; NOV=$R/runs/abl/lib_r1_gate_noval.json
P8=(8123 8133 8124 8134 8125 8135 8126 8136); SW=(http://127.0.0.1:36102 http://127.0.0.1:36112); k=0
run() {  # run NAME SPLIT ENVVARS...
  local name=$1 split=$2; shift 2
  env "$@" SCIWORLD_URL=${SW[$((k%2))]} $T run_eval.py --env sciworld --split $split --method meta_replay --max-steps 30 --workers 2 --run-id abl/$name --llm-url http://127.0.0.1:${P8[$((k%8))]}/v1 2>&1 | grep -E "SUMMARY|Traceback" | sed "s|^|$name |" &
  k=$((k+1))
}
# A1 constraints on the full system
for m in hand learned none; do run A1_shield_${m}_test test PATCHES=$FULL META_SHIELD=$m LCH_TAU=0.9761; done
# A2 no validation (round 1)
run A2_r1_noval_test test PATCHES=$NOV META_SHIELD=both LCH_TAU=0.9761
# A3 procedure components on LCH-only
for a in typedemo nolift nodecide norepair nofinish noground noloop nogoto nodecide,norepair,nofinish; do
  n=${a//,/+}; [ "$a" = nodecide,norepair,nofinish ] && n=nollm
  run A3_${n}_test test ABL=$a META_SHIELD=both LCH_TAU=0.9761
done
# B1 tau on full (held-out), B2 demo fraction on LCH-only (held-out)
for t in 0.90 0.95 0.99 0.995; do run B1_tau${t}_ho train_ho2all PATCHES=$FULL META_SHIELD=both LCH_TAU=$t LOO_DEMO=1; done
for f in 0.1 0.25 0.5; do run B2_frac${f}_ho train_ho2all DEMO_FRAC=$f META_SHIELD=both LCH_TAU=0.9761 LOO_DEMO=1; done
# B4 C4 large-space budget: extend gate and random to B=4
( C4_DIR=$R/runs/repair4 C4_METHODS=gate,random C4_THREADS=16 REPAIR_PORTS=8123,8133,8124,8134,8125,8135,8126,8136 $T evolve_x.py search 4 2>&1 | grep -E "edit runs|SEARCH_DONE|Trace" | sed "s|^|B4 |" ) &
wait; echo ABL_DONE
