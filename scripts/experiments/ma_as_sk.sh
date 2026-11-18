#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# AgentSpec + SkillRL-SB chain for one env, after the base rollouts ma_skillrl_ENV_trainN exist (repo protocol):
#  AgentSpec: rules from the train rollouts -> test.  SkillRL: distill(init) -> train run with bank -> evolve r1 -> test.
# Artifact LLM calls go to the 32k server 8128 (Qwen2.5-7B, also the SkillRL teacher); episodes to TESTPORT.
# usage: ma_as_sk.sh ENV BENCH ENVNAME N TESTPORT [skip_build_ws]
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
ENV=$1; B=$2; EN=$3; N=$4; PORT=$5; A=$PINPATCH_ROOT/runs/ma_art; export MA_ART=$A
P=$PINPATCH_ENVS/ma/bin/python; M=$PINPATCH_ROOT/models/Qwen2.5-7B-Instruct; S=$PINPATCH_ROOT/scripts
L=../MetaAct/logs/runs; [ -e $L/skillrl_${B}_collect ] || ln -s ma_skillrl_${ENV}_train$N $L/skillrl_${B}_collect
[ -f $A/agentspec_${B}_rules.json ] || $P metaact_ag.py raw agentspec_gen_rules $ENV 8128 -- --bench $B --out $A/agentspec_${B}_rules.json 2>&1 | tail -3
[ -s $A/skillrl_${B}.jsonl ] || $P metaact_ag.py raw skillrl_distill $ENV 8128 -- --init --from logs/runs/ma_skillrl_${ENV}_train$N --bank $A/skillrl_${B}.jsonl --env $EN --teacher $M 2>&1 | tail -3
bash $S/ma_run.sh agentspec $ENV test $PORT - --agentspec_rules $A/agentspec_${B}_rules.json &
( $P metaact_ag.py $ENV train$N ma_skillrl_${ENV}_val $PORT -- --baseline skillrl --skillrl_bank $A/skillrl_${B}.jsonl 2>&1 | grep -E "success_rate|Trace"
  $P metaact_ag.py raw skillrl_distill $ENV 8128 -- --evolve --round 1 --from logs/runs/ma_skillrl_${ENV}_val --bank $A/skillrl_${B}.jsonl --env $EN --teacher $M 2>&1 | tail -3
  bash $S/ma_run.sh skillrl $ENV test $PORT - --skillrl_bank $A/skillrl_${B}.jsonl ) &
wait; echo AS_SK_DONE $ENV
