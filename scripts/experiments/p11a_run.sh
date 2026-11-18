#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P1-1 on ALFWorld (pre-registered PROGRESS.md 2026-11-18): evo base -> replay failures under 7 configs -> evolve + gate
# -> random -> held-out base / evolved / random. Qwen serial vLLM on GPU0 (port 8123) for meta_alf's stuck fallback.
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0; E=$PINPATCH_ENVS; T=$E/tc/bin/python
cd $R
tmux has-session -t pp-vllm0 2>/dev/null || tmux new-session -d -s pp-vllm0 "VLLM_UTIL=0.6 VLLM_KV=1610612736 bash scripts/vllm.sh 0 8123 serial > runs/vllm/a40_8123.log 2>&1"
for k in $(seq 90); do curl -s localhost:8123/v1/models | grep -q Qwen && break; sleep 10; done; echo VLLM_UP
U=http://127.0.0.1:8123/v1; cd code/eval
restart_alf() {  # restart_alf PORT SESSION: the ALFWorld server copies a 35 MB planner library per reset and keeps every
  # copy open until it exits; tmp goes to RAM (/dev/shm) and the server is restarted before every stage / retry
  local port=$1 ses=$2; tmux kill-session -t $ses 2>/dev/null; sleep 3; rm -rf /dev/shm/alf_$port; mkdir -p /dev/shm/alf_$port
  tmux new-session -d -s $ses "export TMPDIR=/dev/shm/alf_$port ALFWORLD_DATA=$R/data/alfworld PATH=$E/alf/bin:\$PATH; exec $E/alf/bin/alfworld --host 127.0.0.1 --port $port > $R/scripts/logs/alf_server_$port.log 2>&1"
  for k in $(seq 60); do curl -s -XPOST localhost:$port/create >/dev/null 2>&1 && return 0; sleep 5; done; echo "ALF server $port did not start"
}
need() {  # need SPLIT RUN_ID CMD...: rerun (resumable) until all ids are done, at most 3 tries; else abort
  local sp=$1 rid=$2; shift 2; local n=$($T -c "import json; print(len(json.load(open('$R/data/splits/alfworld_$sp.json'))))")
  local lg=$R/scripts/logs/run_$(echo $rid | tr / _).log
  for t in 1 2 3 4 5; do  # chunks of 200 games per server life (each reset leaks a 35 MB file until the server exits)
    for lim in $(seq 200 200 $n) $n; do restart_alf 36103 pp-alf; "$@" --split $sp --run-id $rid --limit $lim >> $lg 2>&1; done
    grep -E "SUMMARY" $lg | tail -1
    [ "$(cat $R/runs/$rid/episodes.jsonl 2>/dev/null | wc -l)" -ge $n ] && return 0; echo "RETRY $rid try $t"; sleep 30; done
  echo "ABORT: $rid incomplete ($(cat $R/runs/$rid/episodes.jsonl 2>/dev/null | wc -l)/$n)"; exit 1
}
RUN() { $T run_eval.py --env alfworld --method meta_alf --max-steps 30 --workers 8 --llm-url $U "$@"; }
set -o pipefail
[ -f $R/data/splits/alfworld_train_evo.json ] || $T alf_evolve.py splits
need train_evo p11a/evo_base RUN
for t in 1 2 3; do restart_alf 36103 pp-alf; $T alf_evolve.py replay p11a/evo_base $U 2>&1 | tail -3 && break; echo "RETRY replay $t"; sleep 30; done
restart_alf 36103 pp-alf; $T alf_evolve.py evolve p11a/evo_base $U 2>&1 | tail -3 || { echo "ABORT: evolve failed"; exit 1; }
$T alf_evolve.py random
need train_ho p11a/ho_base RUN
ALF_PATCHES=$R/runs/p11a/patches.json need train_ho p11a/ho_evolved RUN
ALF_PATCHES=$R/runs/p11a/patches_random.json need train_ho p11a/ho_random RUN
echo P11A_HO_DONE
