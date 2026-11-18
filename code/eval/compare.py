"""compare.py ENV RUN_ID [RUN_ID ...]: per-task success / mean score / negatives for dev runs."""
import json, sys
from collections import defaultdict
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
t = json.load(open(f"{R}/data/splits/{sys.argv[1]}.json")).get("task", {})
for rid in sys.argv[2:]:
    rs = [json.loads(l) for l in open(f"{R}/runs/{rid}/episodes.jsonl")]
    by = defaultdict(list)
    for r in rs: by[t.get(str(r["id"]), "all")].append(r)
    print("==", rid, "succ", sum(r["success"] for r in rs), "/", len(rs), "mean", round(sum(r.get("score", 0) for r in rs) / len(rs), 1))
    for k, v in sorted(by.items()):
        sc = [r.get("score", 0) for r in v]
        print("  %-40s %d/%-2d mean %6.1f neg %d steps %.0f" % (k, sum(r["success"] for r in v), len(v), sum(sc) / len(sc), sum(s < 0 for s in sc), sum(r["steps"] for r in v) / len(v)))
