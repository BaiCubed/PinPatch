#!/bin/bash
# usage: scripts/setup.sh {envs|agentgym|data|model|link|all}
# Creates the Python environments under $PINPATCH_ENVS with micromamba (must be on PATH), fetches AgentGym and the
# environment data, downloads Qwen2.5-7B-Instruct, and links the shipped artifacts to where the scripts expect them.
set -euo pipefail
source "$(dirname "$0")/config.sh"; R=$PINPATCH_ROOT; E=$PINPATCH_ENVS; A=$R/code/AgentGym
AGENTGYM_COMMIT=3ef9235d23e68e7c2920c5422ad957dc8ced5c6c
mk() { [ -x $E/$1/bin/python ] || micromamba create -y -q -p $E/$1 -c conda-forge "${@:2}" "pip<26.2"; }
step() {
case $1 in
envs)
  # main: vLLM serving, probes (constraint head, gate), SFT. Versions of our last runs: torch 2.14.0, vllm 0.33.0,
  # transformers 5.20.2, peft 0.21.2, trl 1.17.1, numpy 2.3.5 (earlier runs used older vLLM; outputs are deterministic
  # per server, not across library versions)
  mk main python=3.11; $E/main/bin/pip install vllm transformers peft trl numpy pandas scikit-learn
  # tc: controllers and baselines (clients only) + the in-process TextCraft environment
  mk tc python=3.11; $E/tc/bin/pip install numpy gymnasium fastapi uvicorn requests huggingface_hub pyyaml
  (cd $R/code/pinpatch-textcraft && PATH=$E/tc/bin:$PATH bash setup.sh)  # ADaPT TextCraft code (MIT) + 860 recipes
  # sci / alf: the AgentGym ScienceWorld (scienceworld 1.2.3) and ALFWorld (alfworld 0.3.3) servers
  mk sci python=3.8 openjdk=17; $E/sci/bin/pip install -e $A/agentenv-sciworld
  mk alf python=3.9; $E/alf/bin/pip install --only-binary=:all: "spacy==3.7.5"; $E/alf/bin/pip install alfworld==0.3.3
  $E/alf/bin/pip uninstall -y opencv-python || true; $E/alf/bin/pip install -e $A/agentenv-alfworld
  # ma: baselines from the MetaAct code release (only needed for those rows; see README)
  mk ma python=3.11; $E/ma/bin/pip install torch "PyYAML>=6.0.1" "numpy>=1.26" "openai>=1.40.0" python-dotenv requests tqdm \
    matplotlib "sentence-transformers>=3.0.1" gymnasium
  echo "WebShop: see scripts/experiments/ws_setup2.sh and ws_setup3.sh (python 3.8, faiss 1.7, JDK 11, 1000-product index)" ;;
agentgym)
  [ -d $A ] || git clone https://github.com/WooooDyy/AgentGym.git $A
  git -C $A checkout -q $AGENTGYM_COMMIT
  # one read-only route (/valid_actions) used only by the MetaAct-code baselines; it does not change the environment
  git -C $A apply --directory=agentenv-sciworld $R/patches/agentgym_sciworld_valid_actions.patch || echo "patch already applied?" ;;
data)
  mkdir -p $R/data/agentgym $R/data/alfworld/zips
  for f in alfworld sciworld textcraft webshop; do
    curl -fsSL -o $R/data/agentgym/${f}_test.json https://huggingface.co/datasets/AgentGym/AgentEval/resolve/main/${f}_test.json; done
  cd $R/data/alfworld/zips
  for f in json_2.1.1_pddl.zip json_2.1.1_tw-pddl.zip json_2.1.1_json.zip; do
    curl -fsSL -C - -o $f https://github.com/alfworld/alfworld/releases/download/0.2.2/$f; done
  python3 -c "import zipfile,glob; [zipfile.ZipFile(f).extractall('..') for f in glob.glob('*.zip')]"
  echo "SFT baseline only: download AgentGym/AgentTraj-L into $R/data/AgentTraj-L (huggingface datasets)" ;;
model)
  $E/tc/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-7B-Instruct', local_dir='$R/models/Qwen2.5-7B-Instruct'))" ;;
link)  # shipped artifacts -> the run directories the frozen configurations read
  mkdir -p $R/runs/lch $R/runs/phase2_ag_base
  ln -sfn $R/artifacts/lch/v1.pt $R/runs/lch/v1.pt; ln -sfn $R/artifacts/lch/v2e.pt $R/runs/lch/v2e.pt
  ln -sfn $R/artifacts/textcraft/merged $R/runs/phase2_ag_base/merged ;;
all) for s in agentgym envs data model link; do step $s; done ;;
esac; }
step ${1:-all}; echo "SETUP_DONE ${1:-all}"
