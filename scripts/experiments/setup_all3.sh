#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# Rebuild PinPatch on a fresh server. Everything lives under $PINPATCH_ROOT.
# Never overwrite this file while a step is running (bash reads scripts lazily); copy to a new name.
# usage: setup_all3.sh {micromamba|bigenv|data|smallenvs|model}
source $PINPATCH_ROOT/scripts/config.sh
R=$PINPATCH_ROOT; E=$PINPATCH_ENVS; A=$R/code/AgentGym
set -euo pipefail
case $1 in
micromamba)
  [ -x micromamba ] || { curl -fsSL -o micromamba https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64; chmod +x micromamba; } ;;
bigenv)
  if [ ! -x $E/metaact/bin/python ]; then micromamba create -y -q -p $E/metaact -c conda-forge python=3.11 "pip<26.2"; fi
  $E/metaact/bin/pip install --default-timeout=60 --retries 10 vllm transformers peft trl numpy pandas
  $E/metaact/bin/pip freeze > $R/requirements.lock
  $E/metaact/bin/python -c "import torch,vllm;print(torch.__version__,torch.version.cuda,vllm.__version__)" ;;
data)
  cd $R/code
  if [ ! -d AgentGym/agentenv-sciworld ]; then  # GitHub is slow here: fetch only the env dirs we use
    rm -rf AgentGym; GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse https://github.com/WooooDyy/AgentGym.git
    git -C AgentGym sparse-checkout set agentenv agentenv-textcraft agentenv-sciworld agentenv-alfworld
  fi
  git -C AgentGym log -1 --format=%H
  cd $R/data/agentgym
  for f in alfworld sciworld textcraft webshop; do curl -fsSL -m 120 -o ${f}_test.json https://hf-mirror.com/datasets/AgentGym/AgentEval/resolve/main/${f}_test.json; done
  cd $R/code; python3 -c "import zipfile;zipfile.ZipFile('$R/docs/pinpatch-textcraft.zip').extractall('.')"
  D=$R/data/alfworld; BASE=https://github.com/alfworld/alfworld/releases/download/0.2.2
  mkdir -p $D/zips; cd $D/zips
  for f in json_2.1.1_pddl.zip json_2.1.1_tw-pddl.zip json_2.1.1_json.zip; do
    size=$(curl -sIL -m 30 $BASE/$f | grep -i "^content-length" | tail -1 | tr -d "\r" | cut -d" " -f2)
    for t in $(seq 1 300); do
      curl -sSL --speed-time 60 --speed-limit 1000 -m 900 -C - -o $f $BASE/$f || true
      [ "$(stat -c %s $f 2>/dev/null)" = "$size" ] && break; sleep 5
    done
    echo "$f size=$size got=$(stat -c %s $f)"
  done
  python3 -c "
import zipfile,glob
for f in glob.glob('*.zip'):
    z=zipfile.ZipFile(f); print(f, z.testzip()); z.extractall('..')
" ;;
smallenvs)
  until [ -d $A/agentenv-sciworld ]; do sleep 10; done
  if [ ! -x $E/tc/bin/python ]; then micromamba create -y -q -p $E/tc -c conda-forge python=3.11 "pip<26.2"; fi
  $E/tc/bin/pip install -q numpy
  $E/tc/bin/pip install -q gymnasium fastapi uvicorn requests huggingface_hub
  (cd $R/code/pinpatch-textcraft && PATH=$E/tc/bin:$PATH bash setup.sh)
  if [ ! -x $E/sci/bin/python ]; then micromamba create -y -q -p $E/sci -c conda-forge python=3.8 openjdk=17 "pip<26.2"; fi
  $E/sci/bin/pip install -q -e $A/agentenv-sciworld
  if [ ! -x $E/alf/bin/python ]; then micromamba create -y -q -p $E/alf -c conda-forge python=3.9 "pip<26.2"; fi
  $E/alf/bin/pip install -q --only-binary=:all: "spacy==3.7.5"
  $E/alf/bin/pip install -q alfworld==0.3.3
  $E/alf/bin/pip uninstall -y -q opencv-python || true
  $E/alf/bin/pip install -q -e $A/agentenv-alfworld ;;
model)
  until [ -x $E/tc/bin/python ]; do sleep 10; done
  until $E/tc/bin/python -c "import huggingface_hub" 2>/dev/null; do sleep 10; done
  $E/tc/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$R/models/Qwen2.5-7B-Instruct', max_workers=8)
print(p)"
  cd $R/models/Qwen2.5-7B-Instruct && sha256sum *.safetensors > $R/scripts/qwen_sha256.txt ;;
esac
echo "STEP_DONE $1"
