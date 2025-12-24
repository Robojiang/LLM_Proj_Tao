#!/bin/bash

# 设置遇到错误即停止
set -e

# 激活环境 (根据你的环境路径调整)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate llm

cd /mnt/afs/250010074/llm/LLM_Proj_Tao

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0
# 解决 numexpr 警告，允许使用更多 CPU 核心进行计算
export NUMEXPR_MAX_THREADS=192

echo "🚀 开始运行 LoRA Rank 对比实验..."

# ====================================================
# 实验 1: Rank = 8 (基线)
# ====================================================
echo "Running Experiment 1: LoRA Rank = 8 (Baseline)"
python finetune_t5_raw.py \
    model.lora.r=8 \
    model.lora.lora_alpha=16 \
    wandb.run_name=mt5-lora-r8

# ====================================================
# 实验 2: Rank = 32 (提升容量)
# ====================================================
echo "Running Experiment 2: LoRA Rank = 32"
python finetune_t5_raw.py \
    model.lora.r=16 \
    model.lora.lora_alpha=32 \
    wandb.run_name=mt5-lora-r32

# ====================================================
# 实验 3: Rank = 64 (高容量)
# ====================================================
echo "Running Experiment 3: LoRA Rank = 64"
python finetune_t5_raw.py \
    model.lora.r=32 \
    model.lora.lora_alpha=64 \
    wandb.run_name=mt5-lora-r64

echo "🎉 所有 LoRA 实验运行完毕！请去 WandB 查看 train_loss 和 val_loss 的下降曲线对比。"
