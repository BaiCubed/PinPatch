#!/bin/bash
# Start the AgentGym environment servers used by every run (TextCraft runs in-process and needs no server).
#   ScienceWorld 127.0.0.1:36102   ALFWorld 127.0.0.1:36103   WebShop 127.0.0.1:36104   (each in its own tmux session)
# The ALFWorld server copies a ~35 MB planner library for every reset and keeps it open until it exits, so its TMPDIR is
# in RAM and it should be restarted between long runs (scripts/experiments/p11a_run.sh shows the restart loop).
source "$(dirname "$0")/config.sh"; R=$PINPATCH_ROOT; E=$PINPATCH_ENVS
tmux has-session -t pp-sci 2>/dev/null || tmux new -d -s pp-sci "export PATH=$E/sci/bin:\$PATH; exec $E/sci/bin/sciworld --host 127.0.0.1 --port 36102 > $R/runs/logs/sci_server.log 2>&1"
mkdir -p /dev/shm/pp_alf
tmux has-session -t pp-alf 2>/dev/null || tmux new -d -s pp-alf "export TMPDIR=/dev/shm/pp_alf ALFWORLD_DATA=$R/data/alfworld PATH=$E/alf/bin:\$PATH; exec $E/alf/bin/alfworld --host 127.0.0.1 --port 36103 > $R/runs/logs/alf_server.log 2>&1"
[ -x $E/ws/bin/python ] && { tmux has-session -t pp-ws 2>/dev/null || tmux new -d -s pp-ws "bash $R/scripts/experiments/ws_server.sh > $R/runs/logs/ws_server.log 2>&1"; }
for p in 36102 36103; do for k in $(seq 60); do curl -s -XPOST localhost:$p/create >/dev/null 2>&1 && break; sleep 5; done; done
echo ENV_SERVERS_UP
