#! /bin/bash
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-103-bpe
SAVE_ROOT=./saved_models

model=vqvae_lm_base

#SAVE=${SAVE_ROOT}/soft_tau_0.5_shrink_8_cnn_555_222_clean
SAVE=${SAVE_ROOT}/vqvae_lm_base_wiki103_bpe_0.25_soft_tau_0.8_shrink_8
mkdir -p ${SAVE}

cp $0 ${SAVE}/extract.sh

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

grep ^T-bpe $model/train.codes | cut -f2 > ${data_path}/train.extract.bpe
grep ^C $model/train.codes | cut -f2 > ${data_path}/train.extract.code
grep ^T-bpe $model/valid.codes | cut -f2 > ${data_path}/valid.extract.bpe
grep ^C $model/valid.codes | cut -f2 > ${data_path}/valid.extract.code
grep ^T-bpe $model/test.codes | cut -f2 > ${data_path}/test.extract.bpe
grep ^C $model/test.codes | cut -f2 > ${data_path}/test.extract.code

slang=bpe
tlang=code
python preprocess.py --source-lang ${slang} --target-lang ${tlang} --workers 10 --dataset-impl 'cached' \
      --trainpref ${data_path}/train.extract --validpref ${data_path}/valid.extract --testpref  ${data_path}/test.extract \
        --destdir ${model}/codes_bin --joined-dictionary
