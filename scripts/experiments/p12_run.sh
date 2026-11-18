#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P1-2 WebShop learned purchase constraint (pre-registered PROGRESS.md 2026-11-18). Waits for the lab purchase
# simulation, trains the hidden-state head (GPU1) and the TF-IDF head, serves both, runs held-out train_ho (600) with
# each head and with hidden head + P1-1 patches. base = runs/p11/ho_base (same code, same machine). Test decided after.
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0
T=$PINPATCH_ENVS/tc/bin/python; P=$PINPATCH_ENVS/main/bin/python; cd $R/code/eval
until grep -q SIM_DONE $R/scripts/logs/p12_sim.log; do sleep 60; done
mkdir -p $R/runs/p12
[ -f $R/runs/p12/tfidf.pt ] || CUDA_VISIBLE_DEVICES= $P ws_head.py train tfidf $R/runs/p12/tfidf.pt 2>&1 | grep -v Warning | tail -2
[ -f $R/runs/p12/hidden.pt ] || CUDA_VISIBLE_DEVICES=1 $P ws_head.py train hidden $R/runs/p12/hidden.pt 2>&1 | grep -v Warning | tail -2
tmux has-session -t pp-p12h || tmux new-session -d -s pp-p12h "cd $R/code/eval; CUDA_VISIBLE_DEVICES=1 $P ws_head.py serve $R/runs/p12/hidden.pt 36130 > $R/scripts/logs/p12_svc_hidden.log 2>&1"
tmux has-session -t pp-p12t || tmux new-session -d -s pp-p12t "cd $R/code/eval; CUDA_VISIBLE_DEVICES= $P ws_head.py serve $R/runs/p12/tfidf.pt 36131 > $R/scripts/logs/p12_svc_tfidf.log 2>&1"
for p in 36130 36131; do for k in $(seq 60); do curl -s -XPOST localhost:$p/score -H 'content-type: application/json' -d '{"texts":["x"]}' | grep -q tau && break; sleep 10; done; done
RUN() { $T run_eval.py --env webshop --method meta_ws --max-steps 10 --workers 8 --split train_ho "$@" 2>&1 | grep -E "SUMMARY|Traceback" | tail -2; }
WS_SCORER_URL=http://127.0.0.1:36130 RUN --run-id p12/ho_hidden
WS_SCORER_URL=http://127.0.0.1:36131 RUN --run-id p12/ho_tfidf
WS_PATCHES=$R/runs/p11/patches_FROZEN.json WS_SCORER_URL=http://127.0.0.1:36130 RUN --run-id p12/ho_hidden_p11
echo P12_HO_DONE
