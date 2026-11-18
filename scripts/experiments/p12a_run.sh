#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P1-2 on ALFWorld (pre-registered PROGRESS.md 2026-11-18): lab data on a second ALFWorld server (36113), heads (hidden on
# GPU1, tfidf on CPU), served on 36140/36141; held-out runs share the Qwen fallback server of p11a (GPU0, 8123).
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0; E=$PINPATCH_ENVS
T=$E/tc/bin/python; P=$E/metaact/bin/python; cd $R
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
    for lim in $(seq 200 200 $n) $n; do restart_alf 36113 pp-alf2; "$@" --split $sp --run-id $rid --limit $lim >> $lg 2>&1; done
    grep -E "SUMMARY" $lg | tail -1
    [ "$(cat $R/runs/$rid/episodes.jsonl 2>/dev/null | wc -l)" -ge $n ] && return 0; echo "RETRY $rid try $t"; sleep 30; done
  echo "ABORT: $rid incomplete ($(cat $R/runs/$rid/episodes.jsonl 2>/dev/null | wc -l)/$n)"; exit 1
}
until [ -f data/splits/alfworld_train_lab.json ]; do sleep 30; done
cd code/eval; mkdir -p $R/runs/p12a
for lim in 100 200 300 400 500 560; do restart_alf 36113 pp-alf2; ALFWORLD_URL=http://127.0.0.1:36113 LAB_LIMIT=$lim $T alf_head.py data 2>&1 | tail -1; done
grep -q . $R/runs/p12a/lab.jsonl || { echo "ABORT: no lab data"; exit 1; }
[ -f $R/runs/p12a/tfidf.pt ] || CUDA_VISIBLE_DEVICES= $P alf_head.py train tfidf $R/runs/p12a/tfidf.pt 2>&1 | grep -v Warning | tail -2
[ -f $R/runs/p12a/hidden.pt ] || CUDA_VISIBLE_DEVICES=1 $P alf_head.py train hidden $R/runs/p12a/hidden.pt 2>&1 | grep -v Warning | tail -2
tmux has-session -t pp-p12ah || tmux new-session -d -s pp-p12ah "cd $R/code/eval; CUDA_VISIBLE_DEVICES=1 $P alf_head.py serve $R/runs/p12a/hidden.pt 36140 > $R/scripts/logs/p12a_svc_hidden.log 2>&1"
tmux has-session -t pp-p12at || tmux new-session -d -s pp-p12at "cd $R/code/eval; CUDA_VISIBLE_DEVICES= $P alf_head.py serve $R/runs/p12a/tfidf.pt 36141 > $R/scripts/logs/p12a_svc_tfidf.log 2>&1"
for p in 36140 36141; do for k in $(seq 60); do curl -s -XPOST localhost:$p/score -H 'content-type: application/json' -d '{"texts":["x"]}' | grep -q tau && break; sleep 10; done; done
for p in 36140 36141; do curl -s -XPOST localhost:$p/score -H 'content-type: application/json' -d '{"texts":["x"]}' | grep -q tau || { echo "ABORT: head service $p down"; exit 1; }; done
cd $R  # own Qwen fallback server on GPU1 (next to the hidden head), so both GPUs are used
tmux has-session -t pp-vllm1 2>/dev/null || tmux new-session -d -s pp-vllm1 "VLLM_UTIL=0.45 VLLM_KV=1610612736 bash scripts/vllm.sh 1 8124 serial > runs/vllm/a40_8124.log 2>&1"
for k in $(seq 90); do curl -s localhost:8124/v1/models | grep -q Qwen && break; sleep 10; done
curl -s localhost:8124/v1/models | grep -q Qwen || { echo "ABORT: vLLM on GPU1 did not start"; tail -5 runs/vllm/a40_8124.log; exit 1; }
cd code/eval; U=http://127.0.0.1:8124/v1
RUN() { ALFWORLD_URL=http://127.0.0.1:36113 $T run_eval.py --env alfworld --method meta_alf --max-steps 30 --workers 8 --llm-url $U "$@"; }
ALF_SCORER_URL=http://127.0.0.1:36140 need train_ho p12a/ho_hidden RUN
ALF_SCORER_URL=http://127.0.0.1:36141 need train_ho p12a/ho_tfidf RUN
until [ -f $R/runs/p11a/patches.json ] && grep -q . $R/runs/p11a/evolve_log.jsonl 2>/dev/null; do sleep 60; done
ALF_PATCHES=$R/runs/p11a/patches.json ALF_SCORER_URL=http://127.0.0.1:36140 need train_ho p12a/ho_hidden_p11 RUN
echo P12A_HO_DONE
