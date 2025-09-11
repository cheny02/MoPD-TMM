#!/bin/bash

cd ..

# custom config
# DATA=/data/yht/data/cl/data
DATA=/data/dataset/wxq
TRAINER=MoPD
DATASET=$1
WEIGHT=$2
WEIGHT2=$3
num_prompts=$4
#CFG=rn50_ep100  # config file\
#CFG=vit_b16_ep25_ctxv1
CFG=$5
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
for SEED in 1 2 3
# for SEED in 1
do
    DIR=output_MoPD/base2new/train_base/${DATASET}/${TRAINER}/${CFG}/num_prompts_${num_prompts}/shots_${SHOTS}_${WEIGHT}_${WEIGHT2}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.W2 ${WEIGHT2} \
        LOSS.num_prompts ${num_prompts}\
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
