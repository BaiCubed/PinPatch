#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# A2: round-1 gate library WITHOUT replay validation (same proposals as runs/repair2/patches_gate_merged.json)
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; cd $R/code/eval; T=$PINPATCH_ENVS/tc/bin/python; mkdir -p $R/runs/abl
for d in repair repair2; do
  C3_DIR=$R/runs/$d $T -c "import evolve,json; p,t,f,b=evolve.proposed('gate',2); json.dump(p,open('$R/runs/abl/noval_$d.json','w')); print('$d',t,f,len(p))"
done
$T -c "import evolve,json; a=json.load(open('$R/runs/abl/noval_repair.json')); b=json.load(open('$R/runs/abl/noval_repair2.json')); m=evolve.merged(a,b); json.dump(m,open('$R/runs/abl/lib_r1_gate_noval.json','w'),indent=1); print('NOVAL', len(m))"
