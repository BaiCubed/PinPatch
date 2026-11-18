#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# A1 (pre-registered PROGRESS.md 2026-11-18): main-table baselines on another backbone, ScienceWorld + ALFWorld test.
# usage: a1_backbone.sh NAME GPU SLOT PHASE      NAME = models/NAME; SLOT 0|1 picks ports; PHASE dcrpms | aceexpel
# Same implementations and settings as the main table (DC / RPMS / ACE through metaact_ag.py as in ma_run.sh / ma_rpms.sh;
# ExpeL through baselines.py as in r3_rerun.sh); only the served model differs. Two serial vLLM servers per GPU (one for
# phi-4), each job talks to one of them; own ScienceWorld / ALFWorld servers per slot. Resumable (finished ids are skipped).
N=$1; G=$2; K=$3; PH=$4
R=$PINPATCH_ROOT; source $PINPATCH_ROOT/scripts/config.sh; export PYTHONHASHSEED=0; E=$PINPATCH_ENVS
T=$E/tc/bin/python; MP=$E/ma/bin/python; M=$R/models/$N; cd $R
PA=$((8130 + 10 * K)); PB=$((8131 + 10 * K)); SP=$((36120 + K)); AP=$((36125 + K)); LG=runs/logs/a1_${N}_$PH; mkdir -p $LG
KV=2147483648; X=""; UT=0.46
[ $N = olmo3_7b ] && { KV=5368709120; X="--chat-template $M/chat_template_fixed.jinja"; UT=0.48; }
[ $N = phi-4 ] && { UT=0.9; PB=$PA; }  # 14B: one server per GPU
vl() { tmux has-session -t pp-a1v$1 2>/dev/null || tmux new -d -s pp-a1v$1 "VLLM_MODEL=$M VLLM_UTIL=$UT VLLM_KV=$KV VLLM_EXTRA=\"$X\" bash scripts/vllm.sh $G $1 serial > $LG/vllm_$1.log 2>&1"
  for k in $(seq 120); do curl -s localhost:$1/v1/models | grep -q "$M" && return 0; sleep 10; done; echo "ABORT: vLLM $1 did not start"; tail -20 $LG/vllm_$1.log; exit 1; }
vl $PA; [ $PB != $PA ] && vl $PB; echo VLLM_UP $PA $PB
tmux kill-session -t pp-a1s$K 2>/dev/null; tmux kill-session -t pp-a1a$K 2>/dev/null; sleep 3
rm -rf /dev/shm/alf_$AP; mkdir -p /dev/shm/alf_$AP
tmux new -d -s pp-a1s$K "export PATH=$E/sci/bin:\$PATH; exec $E/sci/bin/sciworld --host 127.0.0.1 --port $SP > $LG/sci_server.log 2>&1"
tmux new -d -s pp-a1a$K "export TMPDIR=/dev/shm/alf_$AP ALFWORLD_DATA=$R/data/alfworld PATH=$E/alf/bin:\$PATH; exec $E/alf/bin/alfworld --host 127.0.0.1 --port $AP > $LG/alf_server.log 2>&1"
for p in $SP $AP; do for k in $(seq 60); do curl -s -XPOST localhost:$p/create >/dev/null 2>&1 && break; sleep 5; done; done
export LLM_MODEL=$M SCIWORLD_URL=http://127.0.0.1:$SP ALFWORLD_URL=http://127.0.0.1:$AP OPENAI_API_KEY=EMPTY
cd code/eval
ma() {  # ma METHOD ENV SPLIT PORT [args]: one MetaAct-repo baseline run, run id a1_NAME_METHOD_ENV_SPLIT
  local m=$1 e=$2 s=$3 p=$4; shift 4
  $MP metaact_ag.py $e $s a1_${N}_${m}_${e}_$s $p -- --baseline $m "$@" > $R/$LG/${m}_${e}_$s.log 2>&1
  echo "$(date +%T) done $m $e $s: $(wc -l < $R/code/MetaAct/logs/runs/a1_${N}_${m}_${e}_$s/results.jsonl 2>/dev/null) episodes"; }
rpms() {  # same chain as ma_rpms.sh: learn on the TRAIN slice, then test with the memory read-only
  local e=$1 n=$2 p=$3 mem=$R/runs/ma_mem/a1_${N}_rpms_$1.jsonl; mkdir -p $R/runs/ma_mem
  ma rpms $e train$n $p --rpms_memory $mem --rpms_learn; echo "rpms $e memory entries: $(wc -l < $mem)"; ma rpms $e test $p --rpms_memory $mem; }
expel() { EXPEL_TAG=_a1_$N $T baselines.py --env $1 --split test --method expel --max-steps 30 --limit 0 --workers 4 \
  --run-id a1/$N/${1:0:3}_expel_test --llm-url http://127.0.0.1:$2/v1 > $R/$LG/expel_$1.log 2>&1; echo "$(date +%T) done expel $1: $(grep SUMMARY $R/$LG/expel_$1.log)"; }
if [ $PH = dcrpms ]; then
  ( rpms alfworld 140 $PA ) & sleep 20; ( ma dc sciworld test $PA ) & sleep 20
  ( rpms sciworld 30 $PB ) & sleep 20; ( ma dc alfworld test $PB ) &
else
  ( ma ace sciworld test $PA ) & sleep 20; ( expel alfworld $PA ) & sleep 20
  ( ma ace alfworld test $PB ) & sleep 20; ( expel sciworld $PB ) &
fi
wait
tmux kill-session -t pp-a1v$PA 2>/dev/null; tmux kill-session -t pp-a1v$PB 2>/dev/null
tmux kill-session -t pp-a1s$K; tmux kill-session -t pp-a1a$K; rm -rf /dev/shm/alf_$AP
echo A1_DONE $N $PH
