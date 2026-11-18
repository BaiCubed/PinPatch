#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# P1-1 WebShop verified procedure evolution (pre-registered PROGRESS.md 2026-11-18). CPU only (meta_ws has no LLM).
# evo base -> replay failures under the 11 alternative configs -> evolve + replay gate -> random control ->
# held-out (train_ho, 600) base / evolved / random. Test is NOT run here (decided after held-out, per pre-registration).
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; T=$PINPATCH_ENVS/tc/bin/python; export PYTHONHASHSEED=0
tmux has-session -t pp-ws 2>/dev/null || { tmux new-session -d -s pp-ws "bash $R/scripts/ws_server.sh > $R/scripts/logs/ws_server.log 2>&1"; sleep 60; }
cd $R/code/eval
RUN() { $T run_eval.py --env webshop --method meta_ws --max-steps 10 --workers 8 "$@" 2>&1 | grep -E "SUMMARY|Traceback" | tail -2; }
RUN --split train_evo --run-id p11/evo_base
$T ws_evolve.py replay p11/evo_base
$T ws_evolve.py evolve p11/evo_base
$T ws_evolve.py random p11/evo_base
RUN --split train_ho --run-id p11/ho_base
WS_PATCHES=$R/runs/p11/patches.json RUN --split train_ho --run-id p11/ho_evolved
WS_PATCHES=$R/runs/p11/patches_random.json RUN --split train_ho --run-id p11/ho_random
echo P11_HO_DONE
