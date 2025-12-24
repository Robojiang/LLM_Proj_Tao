#!/bin/bash

# 设置遇到错误即停止，防止一个挂了后面接着跑浪费时间
set -e

cd /mnt/afs/250010074/llm/LLM_Proj_Tao

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate llm

# 指定使用的 GPU，例如使用 0 号卡
export CUDA_VISIBLE_DEVICES=0

echo "🚀 开始运行自动化实验脚本..."

# ====================================================
# 第一组：RNN 基线与注意力机制对比 (RNN Attention Ablation)
# 目的：对比 Dot, General (Multiplicative), Additive 的效果
# ====================================================

echo "Running RNN Experiment 1: Dot Product Attention"
python train.py \
    model=rnn \
    model.attn=dot 

echo "Running RNN Experiment 2: General (Multiplicative) Attention"
python train.py \
    model=rnn \
    model.attn=general \

echo "Running RNN Experiment 3: Additive Attention (Expected Best)"
python train.py \
    model=rnn \
    model.attn=additive \

# ====================================================
# 第二组：RNN 训练策略对比 (Teacher Forcing)
# 目的：对比 Teacher Forcing = 1.0 (默认) vs 0.8 (Scheduled Sampling)
# 注意：上面的 RNN Exp 3 其实就是 TF=1.0 的对照组
# ====================================================

echo "Running RNN Experiment 4: Additive Attention with Scheduled Sampling (TF=0.5)"
python train.py \
    model=rnn \
    model.attn=additive \
    model.teacher_forcing=0.5 \

echo "Running RNN Experiment 4: Additive Attention with Scheduled Sampling (TF=0.0)"
python train.py \
    model=rnn \
    model.attn=additive \
    model.teacher_forcing=0 \

echo "🎉 所有实验运行完毕！请去 WandB 查看曲线，并运行 eval.py 生成最终 BLEU 分数。"