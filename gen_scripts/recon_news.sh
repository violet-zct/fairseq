#! /bin/bash
source activate py36

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

#DATA=../data/wikitext-clean-bpe
DATA=/checkpoint/chuntinz/work/data/data-bin/20news
SAVE_ROOT=/checkpoint/chuntinz/work/fairseq/saved_models

models=( pretrain_doc19_oft_tau_15_shrink_4_window_3_deconv_8192_exp_10k pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_8192_exp_10k_new_conv pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_8192_exp_10k pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_16384_exp_10k pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_16384_exp_10k pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_32768_exp_10k pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_32768_exp_10k )
stride_first=( 1 0 1 1 1 0 1 )

models=( pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_8192_exp_10k_stride_later pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_16384_exp_10k_stride_later pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_32768_exp_10k_stride_later pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_65536_exp_10k_stride_later pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_8192_exp_10k pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_16384_exp_10k pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_32768_exp_10k pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_65536_exp_10k_stride_later )

models=( pretrain_doc_16384_4_no_mea_no_explore pretrain_doc_32768_4_no_mea pretrain_doc_32768_4_no_mea_no_exp pretrain_doc_32768_4_no_mea_no_exp_tau_10_sample_5 pretrain_doc_32768_4_no_mea_win_3_no_explore pretrain_doc19_soft_tau_10_shrink_4_chunk_512_deconv_32768_no_exp pretrain_doc_65536_4_no_mea_no_explore pretrain_doc19_soft_tau_15_shrink_4_window_3_deconv_65536_exp_10k_stride_later pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_65536_exp_10k_stride_later )
#for model in "${models[@]}"

models=( pretrain_c0.25_no_shard_doc19_soft_tau_15_chunk_512_32768_exp_10k pretrain_doc19_soft_tau_10_shrink_4_chunk_512_deconv_32768_no_exp pretrain_doc_32768_4_no_mea_no_exp_tau_10_sample_5 pretrain_c0.1_doc19_soft_tau_15_chunk_512_32768_exp_10k pretrain_doc_32768_4_no_mea pretrain_c0.25_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k pretrain_c0.25_doc19_soft_tau_15_chunk_256_65536_no_shard_exp_10k pretrain_c0.1_no_ema_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k )

models=( pretrain_c0.25_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k pretrain_c0.25_doc19_soft_tau_15_chunk_256_65536_no_shard_exp_10k pretrain_c0.1_no_ema_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k )

models=( pretrain_c0.25_doc19_soft_tau_15_chunk_256_65536_no_shard_exp_10k )
#models=( pretrain_c0.25_no_shard_doc19_soft_tau_15_chunk_512_32768_exp_10k )

#models=( pretrain_doc19_soft_tau_15_shrink_4_chunk_512_deconv_65536_exp_10k_stride_later )
#models=( pretrain_c0.1_doc19_soft_tau_15_chunk_512_32768_no_shard_exp_10k pretrain_c0.1_doc19_soft_tau10_s5_chunk_512_32768_no_shard_exp_10k pretrain_c0.25_doc19_hard_chunk_512_32768_exp_10k  pretrain_c0.25_doc19_soft_tau_10_sample_1_chunk_512_32768 pretrain_c0.1_doc19_soft_tau_15_chunk_512_65536_no_shard_exp_10k pretrain_c0.1_doc19_soft_tau10_s5_chunk_512_65536_no_shard_exp_10k )

for i in ${!models[*]}
do  
    model=${models[$i]}
    #use_stride_first=${stride_first[$i]}

    echo ${model}
    SAVE=${SAVE_ROOT}/${model}

    cp $0 ${SAVE}/reconstruct.sh

    CUDA_VISIBLE_DEVICES=1 python -u vqvae_extract_eval.py ${DATA} \
    --vqvae-path $SAVE/checkpoint_last.pt --code-extract-strategy topk \
    --eval-task reconstruct \
    --task VQVAE_language_modeling \
    --max-tokens 5072 --shard-id 0 \
    --results-path $SAVE --remove-bpe "@@ " \
    --gen-subset 'train' --dataset-impl mmap \
    --skip-invalid-size-inputs-valid-test \
    --log-format simple --log-interval 500 | tee ${SAVE}/reconstruct_log.txt
    
    #echo $i
    #if [ $i = 0 ];then exit;fi
done


    #--model-overrides "{'use_stride_first':${use_stride_first}}" \
