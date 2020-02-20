#! /bin/bash
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

#DATA=../data/wikitext-clean-bpe
#DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-103-bpe
DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-clean-bpe
SAVE_ROOT=/checkpoint/chuntinz/work/fairseq/saved_models

model=vqvae_lm_base

#SAVE=${SAVE_ROOT}/soft_tau_5_shrink_8_cnn_555_222_full_conv_nobos
SAVE=${SAVE_ROOT}/soft_tau_2_shrink_8_cnn_333_222_full_conv_nobos
#SAVE=${SAVE_ROOT}/soft_tau_2_shrink_8_cnn_555_222_full_conv_nobos
#SAVE=${SAVE_ROOT}/soft_tau_2_shrink_8_cnn_456_345_234_222_full_conv_nobos
#SAVE=${SAVE_ROOT}/soft_tau_2_shrink_8_cnn_345_222_full_conv_nobos
SAVE=${SAVE_ROOT}/soft_tau_2_sanity_check_30k_kernel_3_full_conv_nobos
PRIOR=${SAVE_ROOT}/lm_prior_soft_tau_2_sanity_check_30k_kernel_3_full_conv_nobos
mkdir -p ${SAVE}

cp $0 ${SAVE}/reconstruct.sh

    #--sampling --sampling-topk 10 \
CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt \
    --prior-path ${PRIOR}/checkpoint_best.pt \
    --eval-task sampling \
    --task VQVAE_language_modeling \
    --tokens-per-sample 1024 --max-tokens 3072 \
    --sample-break-mode 'eos' \
    --results-path $SAVE --remove-bpe "@@ " \
    --valid-subset 'valid' --dataset-impl cached \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/reconstruct_log.txt

