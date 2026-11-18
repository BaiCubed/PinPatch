#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
# WebShop artifacts with Qwen2.5-7B on the 32k server (8128): AgentSpec rules, SkillRL bank (init from train rollouts)
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval
P=$PINPATCH_ENVS/ma/bin/python; A=$PINPATCH_ROOT/runs/ma_art; M=$PINPATCH_ROOT/models/Qwen2.5-7B-Instruct
export MA_ART=$A
$P metaact_ag.py raw agentspec_gen_rules webshop 8128 -- --bench ws --out $A/agentspec_ws_rules.json 2>&1 | tail -5
$P metaact_ag.py raw skillrl_distill webshop 8128 -- --init --from logs/runs/ma_skillrl_webshop_train20 --bank $A/skillrl_ws.jsonl --env WebShop --teacher $M 2>&1 | tail -5
echo BUILD_WS_DONE
