#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# WebShop data from a third-party HF copy (user-approved 2026-11-04). sha256 of the big files matched across
# three independent uploaders; small files are checked against the AgentGym repo copies afterwards.
D=$PINPATCH_ROOT/data/webshop; cd $D
B=https://hf-mirror.com/datasets/YWZBrandon/webshop-data/resolve/main
for f in items_human_ins.json items_ins_v2_1000.json items_shuffle_1000.json items_ins_v2.json items_shuffle.json; do
  for t in $(seq 1 50); do curl -sSL -C - -o $f --retry 3 -m 3600 $B/$f && break; sleep 10; done
  echo "$f $(stat -c %s $f) $(sha256sum $f | cut -c1-16)"
done
echo WSDATA_DONE
