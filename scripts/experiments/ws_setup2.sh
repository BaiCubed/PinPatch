#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# WebShop (AgentGym agentenv-webshop) env: python3.8 + faiss 1.7 + openjdk 11, pinned requirements,
# spaCy en_core_web_lg 3.3.0 (hf-mirror revision 1d03e651c4e6, sha256 = the one AgentGym pins), 1000-product data
# from the AgentGym repo itself (AgentGym's server uses num_products=1000), pyserini indexes.
set -e
source $PINPATCH_ROOT/scripts/config.sh
R=$PINPATCH_ROOT; E=$PINPATCH_ENVS; W=$R/code/AgentGym-ws/agentenv-webshop
if [ ! -x $E/ws/bin/python ]; then micromamba create -y -q -p $E/ws -c conda-forge python=3.8 faiss-cpu=1.7 openjdk=11 "pip<26.2"; fi
P=$E/ws/bin/pip
# wheel first (big; ranged parallel download), verify sha
WH=$R/data/webshop/en_core_web_lg-3.3.0-py3-none-any.whl
if [ "$(sha256sum $WH 2>/dev/null | cut -d" " -f1)" != 6ce19d37dfe5280400f80a5954d41afca10cbc742b97bfcf4b0e452b6eb24273 ]; then
  U=https://hf-mirror.com/spacy/en_core_web_lg/resolve/1d03e651c4e6/en_core_web_lg-any-py3-none-any.whl
  T=400651786; N=24; S=$(( (T+N-1)/N )); mkdir -p $R/data/webshop/whl; cd $R/data/webshop/whl
  for i in $(seq 0 $((N-1))); do s=$((i*S)); e=$((s+S-1)); [ $e -ge $T ] && e=$((T-1)); f=$(printf w%02d $i)
    ( for t in $(seq 1 30); do h=$(stat -c %s $f 2>/dev/null || echo 0); [ $h -eq $((e-s+1)) ] && break
        curl -sSL -m 1800 -r $((s+h))-$e "$U" >> $f || true; sleep 2; done ) & done; wait
  cat w?? > $WH; cd ..; rm -rf whl
  [ "$(sha256sum $WH | cut -d" " -f1)" = 6ce19d37dfe5280400f80a5954d41afca10cbc742b97bfcf4b0e452b6eb24273 ] || { echo WHEEL_SHA_BAD; exit 1; }
fi
echo WHEEL_OK
cd $W/webshop
grep -v "^gradio" requirements.txt > ${TMPDIR:-/tmp}/ws_req.txt  # gradio (web demo only, unpinned) sends pip into endless backtracking
$P install -q -r ${TMPDIR:-/tmp}/ws_req.txt
$P install -q -U "Werkzeug>=2,<3" "mkl>=2021,<2022" "typing_extensions<4.6.0" "gym==0.23.1"
$P install -q $WH
mkdir -p data; cp $R/data/webshop/items_shuffle_1000.json.agentgym data/items_shuffle_1000.json; cp $R/data/webshop/items_ins_v2_1000.json.agentgym data/items_ins_v2_1000.json
cp $R/data/webshop/items_human_ins.agentgym.json data/items_human_ins.json
cd search_engine; mkdir -p resources resources_100 resources_1k resources_100k
PYTHONPATH=$W/webshop $E/ws/bin/python convert_product_file_format.py
mkdir -p indexes; PATH=$E/ws/bin:$PATH JAVA_HOME=$E/ws bash ./run_indexing.sh > $R/scripts/logs/ws_index.log 2>&1
cd $W; $P install -q -e .; $P uninstall -y -q numpy; $P install -q numpy
$E/ws/bin/python -c "import spacy; spacy.load('en_core_web_lg'); import agentenv_webshop; print('WS_IMPORT_OK')"
echo WSSETUP_DONE
