#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# Remaining WebShop setup steps after ws_setup2.sh installed the packages: data, pyserini indexes, package install.
# JAVA_HOME must point at the conda JDK (lib/jvm) for pyjnius.
set -e
source $PINPATCH_ROOT/scripts/config.sh
R=$PINPATCH_ROOT; E=$PINPATCH_ENVS; W=$R/code/AgentGym-ws/agentenv-webshop
P=$E/ws/bin/pip; export JAVA_HOME=$E/ws/lib/jvm; export PATH=$E/ws/bin:$JAVA_HOME/bin:$PATH
cd $W/webshop
mkdir -p data; cp $R/data/webshop/items_shuffle_1000.json.agentgym data/items_shuffle_1000.json; cp $R/data/webshop/items_ins_v2_1000.json.agentgym data/items_ins_v2_1000.json
cp $R/data/webshop/items_human_ins.agentgym.json data/items_human_ins.json
cd search_engine; mkdir -p resources resources_100 resources_1k resources_100k
PYTHONPATH=$W/webshop $E/ws/bin/python convert_product_file_format.py
mkdir -p indexes; bash ./run_indexing.sh > $R/scripts/logs/ws_index.log 2>&1
cd $W; $P install -q -e .; $P uninstall -y -q numpy; $P install -q numpy
$E/ws/bin/python -c "import spacy; spacy.load('en_core_web_lg'); import agentenv_webshop; print('WS_IMPORT_OK')"
echo WSSETUP_DONE
