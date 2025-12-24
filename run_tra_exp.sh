#!/bin/bash

# 遇到错误立即停止
set -e

# 设置 GPU (根据实际情况修改)
export CUDA_VISIBLE_DEVICES=0
# 解决 numexpr 警告，允许使用更多 CPU 核心进行计算
export NUMEXPR_MAX_THREADS=192

cd /mnt/afs/250010074/llm/LLM_Proj_Tao

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate llm

echo "🚀 开始运行 Transformer 全面评估实验..."

# ============================================================
# Group 1: 架构消融 (Architecture Ablation)
# 变量: pos_embedding_type, norm_type
# ============================================================

echo ">>> [Group 1] Running Baseline: Absolute Pos + LayerNorm"
python train.py \
    model=transformer \
    model.pos_embedding_type=absolute \
    model.norm_type=layernorm \

echo ">>> [Group 1] Running Exp: Relative Pos + LayerNorm"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=layernorm \

echo ">>> [Group 1] Running Exp: Relative Pos + RMSNorm (Best Config)"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \


# ============================================================
# Group 2: Batch Size 敏感性 (Batch Size Sensitivity)
# 变量: train.batch_size
# 基准: Batch=128 (已在上面 Group 1 的第三个实验跑过，这里不再重复跑)
# ============================================================

echo ">>> [Group 2] Running Batch Size = 64"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    train.batch_size=64 \
   
echo ">>> [Group 2] Running Batch Size = 256"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    train.batch_size=256 \
    


# ============================================================
# Group 3: 学习率敏感性 (Learning Rate Sensitivity)
# 变量: train.lr
# 基准: LR=5e-4 (已在 Group 1 跑过)
# ============================================================

echo ">>> [Group 3] Running LR = 5e-5"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    train.lr=0.00005 \

echo ">>> [Group 3] Running LR = 1.5e-4"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    train.lr=0.00015 \


# ============================================================
# Group 4: 模型规模 (Model Scales)
# 变量: d_model, nhead, num_layers, dim_feedforward
# 基准: Base (d=512, h=8, l=6) (已在 Group 1 跑过)
# ============================================================

echo ">>> [Group 4] Running Model Scale: Tiny (d=256, L=3)"
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    model.d_model=256 \
    model.dim_feedforward=1024 \
    model.nhead=4 \
    model.num_encoder_layers=3 \
    model.num_decoder_layers=3 \

echo ">>> [Group 4] Running Model Scale: Big (d=768, L=6)"
# 注意: H100上跑这个没问题，如果显存不够请减小 BatchSize
python train.py \
    model=transformer \
    model.pos_embedding_type=relative \
    model.norm_type=rmsnorm \
    model.d_model=768 \
    model.dim_feedforward=3072 \
    model.nhead=12 \
    model.num_encoder_layers=6 \
    model.num_decoder_layers=6 \

echo "🎉 所有 Transformer 实验运行完毕！"