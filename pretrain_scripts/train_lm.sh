#! /bin/bash
##SBATCH --output=/checkpoint/chuntinz/fairseq/logs/slurm-%A.out
##SBATCH --error=/checkpoint/chuntinz/fairseq/logs/slurm-%A.err
#SBATCH --job-name=transformer.lm.share
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="8.23 EMNLP camera ready"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=470g
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
#vqvae_model='soft_tau_2_sanity_check_30k_kernel_3_full_conv_nobos'
#DATA=/checkpoint/chuntinz/work/fairseq/saved_models/${vqvae_model}/codes_bin
DATA='/checkpoint/chuntinz/work/data/data-bin/doc-ende19-v2'
SAVE_ROOT=/checkpoint/chuntinz/work/fairseq/saved_models
PORT=15213
model=transformer_lm

SAVE=${SAVE_ROOT}/pretrain_lm_doc_mono
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

srun --label python -u train.py ${DATA} \
    --arch ${model} --distributed-port $PORT --distributed-world-size 16 \
    --task language_modeling \
    --criterion label_smoothed_cross_entropy \
    --save-dir $SAVE \
    --seed 1 \
    --max-update 10000000 \
    --warmup-updates 6000 --warmup-init-lr 1e-07 \
    --optimizer adam --lr 0.0003 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --adam-betas '(0.9, 0.98)' \
    --tokens-per-sample 256 --max-tokens 8072 --max-target-positions 1024 \
    --sample-break-mode 'complete_doc' --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    --label-smoothing 0.1 --decoder-normalize-before \
    --keep-last-epochs 5 \
    --dataset-impl mmap \
    --log-format simple --log-interval 500 | tee ${SAVE}/log.txt

