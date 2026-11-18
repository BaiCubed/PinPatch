#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# AgentGym WebShop env server on 127.0.0.1:36104 (python3.8 env ws, conda JDK for pyserini)
E=$PINPATCH_ENVS; export JAVA_HOME=$E/ws/lib/jvm; export PATH=$E/ws/bin:$JAVA_HOME/bin:$PATH
cd $PINPATCH_ROOT/code/AgentGym-ws/agentenv-webshop/webshop
exec $E/ws/bin/webshop --host 127.0.0.1 --port 36104
