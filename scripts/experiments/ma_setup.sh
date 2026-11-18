#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# Python env for the MetaAct repo (methods + baselines) talking to our AgentGym servers; torch CPU wheel for sentence-transformers
set -e
source $PINPATCH_ROOT/scripts/config.sh
E=$PINPATCH_ENVS
if [ ! -x $E/ma/bin/python ]; then micromamba create -y -q -p $E/ma -c conda-forge python=3.11 "pip<26.2"; fi
$E/ma/bin/pip install -q torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu 2>/dev/null || $E/ma/bin/pip install -q torch
$E/ma/bin/pip install -q "PyYAML>=6.0.1" "numpy>=1.26" "openai>=1.40.0" "python-dotenv>=1.0" "requests>=2.32.3" "tqdm>=4.66.5" "matplotlib>=3.9.2" "sentence-transformers>=3.0.1" gymnasium
$E/ma/bin/python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\"); print(\"ST_OK\", m.encode([\"hi\"]).shape)"
echo MA_SETUP_DONE
