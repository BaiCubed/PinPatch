#!/bin/bash
# Smoke test (CPU only, no LLM, about a minute after setup): the frozen TextCraft model on the first 5 test tasks.
# Expected: 5/5 solved. Step counts can differ from the paper run by a step or two: AgentGym's TextCraft builds its crafting
# tree from the recipe files in os.listdir order, which depends on the filesystem (paper run: 2,2,4,2,2 steps).
source "$(dirname "$0")/config.sh"; R=$PINPATCH_ROOT; T=$PINPATCH_ENVS/tc/bin/python
W=$(mktemp -d); cp $R/artifacts/textcraft/merged/model.pkl $R/artifacts/textcraft/merged/FREEZE.json $W/
cd $R/code/eval && TC_LIMIT=5 $T ag_run.py $W eval test mem_analog_gated 0 | tail -3
$T - "$W" <<'PY'
import json, sys
got = json.load(open(sys.argv[1] + '/results/test_mem_analog_gated.json'))
res = {e['id']: (e['won'], e['steps']) for e in got['episodes']}
print('episodes (id: solved, steps)', res, '| paper run steps: 2, 2, 4, 2, 2')
ok = len(res) == 5 and all(w for w, _ in res.values())
print('SMOKE_OK' if ok else 'SMOKE_FAIL (expected all 5 solved)'); sys.exit(0 if ok else 1)
PY
