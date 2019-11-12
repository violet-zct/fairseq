#! /bin/bash
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

#DATA=../data/wikitext-clean-bpe
DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-103-bpe
SAVE_ROOT=./saved_models

model=vqvae_lm_base

SAVE=${SAVE_ROOT}/vqvae_lm_base_wiki103_bpe_0.25_soft_tau_0.8_shrink_8
mkdir -p ${SAVE}

cp $0 ${SAVE}/reconstruct.sh

CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt \
    --eval-task reconstruct \
    --task VQVAE_language_modeling \
    --add-bos-token \
    --tokens-per-sample 1024 --max-tokens 3072 \
    --sample-break-mode 'eos' \
    --results-path $SAVE --remove-bpe "@@ " \
    --valid-subset 'valid' --dataset-impl cached \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/extract_log.txt

