#! /bin/bash
##SBATCH --output=/checkpoint/chuntinz/fairseq/logs/slurm-%A.out
##SBATCH --error=/checkpoint/chuntinz/fairseq/logs/slurm-%A.err
#SBATCH --job-name=doc.mt.65536.v4.pretrain.ft.cb.0.5.full.chunk.256
#SBATCH --partition=priority
#SBATCH --comment="2.6 ICML"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=300g
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH -C volta32gb

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

module load cuda/10.0
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

DATE=`date +%Y%m%d`
vqvae_model_root=/checkpoint/chuntinz/work/fairseq/saved_models
vqvae_model=pretrain_c0.25_no_shard_doc19_soft_tau_15_chunk_512_32768_exp_10k
#vqvae_model=pretrain_c0.1_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k
vqvae_model=pretrain_c0.25_doc19_soft_tau_15_chunk_256_65536_no_shard_exp_10k
vqvae_model_path=${vqvae_model_root}/${vqvae_model}/checkpoint_last.pt
DATA=/checkpoint/chuntinz/work/data/data-bin/final-mt-doc-ende19
SAVE_ROOT=/checkpoint/chuntinz/work/fairseq/mt_saved_models

model="transformer_doc_mt_wmt_en_de"
slang=en
SAVE=${SAVE_ROOT}/mt_v4_sep_65536_4_chunk_256_cb_ft_w0.5_full_chunk_base
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python -u train.py ${DATA} \
     -a ${model} --optimizer adam --lr 0.0005 -s en -t de -c pretrain_dict.en \
     --input-form sep --context-form doc --window-size 0  --context-compress '2,2' \
     --context-model-path ${vqvae_model_path} --encode-code 1 --use-seg-pos-emb 0 --ctx-value-weight 0.5 \
     --sep-attn-share-key-proj 1 --fix-code-book 0 --tokens-per-sample 256 \
     --label-smoothing 0.1 --max-tokens 8192 --code-extract-strategy full \
     --eval-dir ${SAVE}/evals --best-checkpoint-metric "bleu" \
     --max-source-positions 2048 --max-target-positions 1024 \
     --keep-last-epochs 5 --task doc_translation \
     --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --max-update 500000 \
     --warmup-updates 4000 --warmup-init-lr '1e-07' \
     --adam-betas '(0.9, 0.98)' --save-dir ${SAVE} \
     --dataset-impl mmap \
     --log-format simple --log-interval 100 \
     --share-all-embeddings \
    | tee ${SAVE}/log_1.txt

#--save-interval-updates 10\
