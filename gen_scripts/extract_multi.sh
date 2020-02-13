#! /bin/bash
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-clean-bpe
SAVE_ROOT=./saved_models

model=vqvae_lm_base

#SAVE=${SAVE_ROOT}/soft_tau_0.5_shrink_8_cnn_555_222_clean
SAVE=${SAVE_ROOT}/soft_tau_2_shrink_8_cnn_456_345_234_222_full_conv_nobos
mkdir -p ${SAVE}

cp $0 ${SAVE}/extract.sh

sleep 4h
CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt \
    --eval-task code_extract \
    --task VQVAE_language_modeling \
    --tokens-per-sample 1024 --max-tokens 8072 \
    --sample-break-mode 'eos' \
    --results-path $SAVE --remove-bpe "@@ " \
    --valid-subset 'train' --dataset-impl cached \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/extract_log.txt

CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt \
    --eval-task code_extract \
    --task VQVAE_language_modeling \
    --tokens-per-sample 1024 --max-tokens 8072 \
    --sample-break-mode 'eos' \
    --results-path $SAVE --remove-bpe "@@ " \
    --valid-subset 'valid' --dataset-impl cached \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/extract_log.txt

CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt \
    --eval-task code_extract \
    --task VQVAE_language_modeling \
    --tokens-per-sample 1024 --max-tokens 8072 \
    --sample-break-mode 'eos' \
    --results-path $SAVE --remove-bpe "@@ " \
    --valid-subset 'test' --dataset-impl cached \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/extract_log.txt

root_dir="/checkpoint/chuntinz/work/fairseq/saved_models"
model=${SAVE}
data_path="$model/codes"
mkdir -p ${data_path}

rm -rf ${model}/codes_bin

grep ^C $model/train.codes | cut -f2 > ${data_path}/train.extract.code
grep ^C $model/valid.codes | cut -f2 > ${data_path}/valid.extract.code
grep ^C $model/test.codes | cut -f2 > ${data_path}/test.extract.code

python preprocess.py --only-source --workers 10 --dataset-impl 'cached' \
      --trainpref ${data_path}/train.extract.code --validpref ${data_path}/valid.extract.code --testpref  ${data_path}/test.extract.code --destdir ${model}/codes_bin 
