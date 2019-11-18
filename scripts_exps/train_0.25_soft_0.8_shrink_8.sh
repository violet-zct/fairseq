#! /bin/bash
##SBATCH --output=/checkpoint/chuntinz/fairseq/logs/slurm-%A.out
##SBATCH --error=/checkpoint/chuntinz/fairseq/logs/slurm-%A.err
#SBATCH --job-name=vqvae.noanneal
##SBATCH --partition=priority
##SBATCH --comment="8.23 EMNLP camera ready"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320

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
DATA=/checkpoint/chuntinz/work/data/data-bin/wikitext-103-bpe
SAVE_ROOT=/checkpoint/chuntinz/work/fairseq/saved_models

model=vqvae_lm_base
run_name="soft_tau_0.8_shrink_8_cnn_555_222_bpe"
SAVE=${SAVE_ROOT}/$model_${run_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python -u train.py ${DATA} \
    --arch ${model} \
    --task VQVAE_language_modeling \
    --criterion vqvae_label_smoothed_cross_entropy \
    --save-dir $SAVE \
    --seed 1 \
    --tensorboard-logdir ./tb-logs/$run_name \
    --add-bos-token \
    --add-latent-positions 0 \
    --bottom-conv-stride '2,2,2' \
    --bottom-conv-kernel-size '5,5,5' \
    --soft-em 1 --soft-max-temp 5.0 --soft-min-temp 0.8 \
    --soft-temp-anneal-steps 0 --soft-samples 5 \
    --commitment-cost 0.25 \
    --max-update 500000 \
    --warmup-updates 6000 --warmup-init-lr 1e-07 \
    --optimizer adam --lr 0.0003 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --adam-betas '(0.9, 0.98)' \
    --tokens-per-sample 1024 --max-tokens 3072 \
    --sample-break-mode 'eos' --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    --label-smoothing 0.1 \
    --keep-last-epochs 5 \
    --dataset-impl cached \
    --log-format simple --log-interval 500 \
    --share-all-embeddings | tee ${SAVE}/log.txt

