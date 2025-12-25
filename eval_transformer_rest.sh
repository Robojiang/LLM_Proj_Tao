#!/bin/bash

# 遇到错误立即停止
set -e

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

cd /mnt/afs/250010074/llm/LLM_Proj_Tao

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate llm

echo "🚀 开始评估剩余的 Transformer 模型..."

# ============================================================
# Transformer 实验评估 (补充部分)
# ============================================================

# 1. Transformer Big (d=768, L=6)
# 对应 run_tra_exp.sh 中的 Group 4 Big Model
echo ">>> Evaluating Transformer Big (d=768, L=6)"
# 注意：这里假设该实验已经跑完并生成了对应的目录，你需要根据实际生成的目录名修改下面的路径
# 目前根据你的 runs 目录列表，最新的一个是 20251225_124943_transformer_relative_rmsnorm_128_0.0001_768
python eval.py \
    --ckpt ./runs/20251225_124943_transformer_relative_rmsnorm_128_0.0001_768/best.pt \
    --data_dir ./processed_data \
    --test_file test.jsonl \
    --decode beam \
    --beam_size 5 \
    --max_len 50

echo "🎉 剩余 Transformer 评估完成！"
