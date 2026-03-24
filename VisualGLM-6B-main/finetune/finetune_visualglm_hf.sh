#!/bin/bash
# set -euo pipefail

# NUM_WORKERS=1
# NUM_GPUS_PER_WORKER=8
# MP_SIZE=1

# script_path=$(realpath "$0")
# script_dir=$(dirname "$script_path")
# main_dir=$(dirname "$script_dir")

# MODEL_DIR="/home/dengxuan/VisualGLM-6B-main/visualglm-6b"
# MODEL_TYPE="visualglm-6b"

# HOST_FILE_PATH="hostfile_single"

# TRAIN_DATA="./fewshot-data/dataset.json"
# EVAL_DATA="./fewshot-data/dataset.json"
# OUTPUT_DIR="./checkpoints"

# MODEL_ARGS="--model_dir ${MODEL_DIR} \
#     --max_source_length 64 \
#     --max_target_length 256 \
#     --lora_rank 10 \
#     --layer_range 0 14 \
#     --pre_seq_len 4"

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"

# GPT_OPTIONS=" \
#     --experiment-name finetune-${MODEL_TYPE} \
#     --model-parallel-size ${MP_SIZE} \
#     --mode finetune \
#     --train-iters 300 \
#     --resume-dataloader \
#     ${MODEL_ARGS} \
#     --train-data ${TRAIN_DATA} \
#     --valid-data ${EVAL_DATA} \
#     --distributed-backend nccl \
#     --lr-decay-style cosine \
#     --warmup 0.02 \
#     --checkpoint-activations \
#     --save-interval 300 \
#     --eval-interval 10000 \
#     --save ${OUTPUT_DIR} \
#     --split 1 \
#     --eval-iters 10 \
#     --eval-batch-size 8 \
#     --zero-stage 1 \
#     --lr 0.0001 \
#     --batch-size 4 \
#     --skip-init \
#     --fp16 \
#     --use_lora \
#     --save_epoch \
# "

# RUN_CMD="${OPTIONS_NCCL} deepspeed \
#     --master_port 16666 \
#     --hostfile ${HOST_FILE_PATH} \
#     ${main_dir}/finetune_visualglm_hf.py \
#     ${GPT_OPTIONS}"

# echo "${RUN_CMD}"
# eval "${RUN_CMD}"
#上面是多卡训练的一个脚本

#!/bin/bash
set -euo pipefail

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
main_dir=$(dirname "$script_dir")

MODEL_DIR="/home/dengxuan/VisualGLM-6B-main/visualglm-6b"
MODEL_TYPE="visualglm-6b"

TRAIN_DATA="./fewshot-data/dataset.json"
EVAL_DATA="./fewshot-data/dataset.json"
OUTPUT_DIR="./checkpoints"

MODEL_ARGS="--model_dir ${MODEL_DIR} \
    --max_source_length 64 \
    --max_target_length 256 \
    --lora_rank 10 \
    --layer_range 0 14 \
    --pre_seq_len 4"

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"

GPT_OPTIONS=" \
    --experiment-name finetune-${MODEL_TYPE} \
    --model-parallel-size 1 \
    --mode finetune \
    --train-iters 300 \
    --resume-dataloader \
    ${MODEL_ARGS} \
    --train-data ${TRAIN_DATA} \
    --valid-data ${EVAL_DATA} \
    --distributed-backend nccl \
    --lr-decay-style cosine \
    --warmup 0.02 \
    --checkpoint-activations \
    --save-interval 300 \
    --eval-interval 10000 \
    --save ${OUTPUT_DIR} \
    --split 1 \
    --eval-iters 10 \
    --eval-batch-size 8 \
    --zero-stage 1 \
    --lr 0.0001 \
    --batch-size 4 \
    --skip-init \
    --fp16 \
    --use_lora \
    --save_epoch \
"

RUN_CMD="CUDA_VISIBLE_DEVICES=7 ${OPTIONS_NCCL} deepspeed \
    --master_port 16666 \
    --num_gpus 1 \
    ${main_dir}/finetune_visualglm_hf.py \
    ${GPT_OPTIONS}"

echo "${RUN_CMD}"
eval "${RUN_CMD}"
#这部分是单卡训练的脚本