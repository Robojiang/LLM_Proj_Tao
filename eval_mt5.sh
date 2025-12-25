#!/bin/bash

# 遇到错误立即停止
set -e

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0
# 解决 numexpr 警告，允许使用更多 CPU 核心进行计算
export NUMEXPR_MAX_THREADS=192

cd /mnt/afs/250010074/llm/LLM_Proj_Tao

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate llm

echo "🚀 开始评估 mT5 模型..."

# ============================================================
# mT5 实验评估
# ============================================================

# 1. mT5 LoRA (Raw Data) - 20251225_003937
echo ">>> Evaluating mT5 LoRA (Raw Data) - Run 1"
python eval_t5_raw.py \
    --model_path ./runs/20251225_003937_mt5-finetune-raw_google-mt5-small_lora_raw/best_model \
    --data_dir ./data \
    --test_file test.jsonl \
    --num_beams 4 \
    --max_len 80

# 2. mT5 LoRA (Raw Data) - 20251225_035345
echo ">>> Evaluating mT5 LoRA (Raw Data) - Run 2"
python eval_t5_raw.py \
    --model_path ./runs/20251225_035345_mt5-finetune-raw_google-mt5-small_lora_raw/best_model \
    --data_dir ./data \
    --test_file test.jsonl \
    --num_beams 4 \
    --max_len 80

# 3. mT5 LoRA (Raw Data) - 20251225_070549
echo ">>> Evaluating mT5 LoRA (Raw Data) - Run 3"
python eval_t5_raw.py \
    --model_path ./runs/20251225_070549_mt5-finetune-raw_google-mt5-small_lora_raw/best_model \
    --data_dir ./data \
    --test_file test.jsonl \
    --num_beams 4 \
    --max_len 80

echo "🎉 所有 mT5 评估完成！"
