#!/bin/bash
: "${PINPATCH_ROOT:?run: source scripts/config.sh}"
source $PINPATCH_ROOT/scripts/config.sh; cd $PINPATCH_ROOT/code/eval; export CUDA_VISIBLE_DEVICES=5
P=$PINPATCH_ENVS/main/bin/python; D=$PINPATCH_ROOT/runs/lch; G=$PINPATCH_ROOT/runs/lch1_sci_noshield_train
$P lch.py data ${G}_a ${G}_b ${G}_c ${G}_d $D/v1.jsonl 2>&1 | tail -1
$P lch.py feats $D/v1.jsonl $D/v1.npz 2>&1 | grep -v Loading | tail -1
$P lch.py train $D/v1.npz $D/v1.pt 2>&1 | tail -2
echo LCH_TRAIN_DONE
