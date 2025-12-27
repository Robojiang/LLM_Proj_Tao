# LLM Project - Machine Translation (Chinese to English)

本项目实现了基于 RNN、Transformer 和 MT5 (LoRA微调) 的中英机器翻译模型。

## 目录结构
- `best_weight/`: 存放训练好的最佳模型权重。
- `data/`: 训练和测试数据 (原始文本)。
- `processed_data/`: 预处理后的数据 (Tokenized)。
- `inference.py`: **统一评估脚本** (计算 BLEU 分数)。
- `eval.py`: RNN/Transformer 评估脚本 (旧版)。
- `eval_t5_raw.py`: MT5 评估脚本 (旧版)。

## 快速开始 (Evaluation)

使用 `inference.py` 脚本可以对测试集进行批量评估，计算 BLEU 分数并展示样例。

### 1. 默认运行
默认情况下，脚本会评估所有可用模型 (RNN, Transformer, MT5)，并使用贪婪解码 (Greedy Search)。

```bash
python inference.py
```

### 2. 指定模型类型
使用 `--model_type` 参数指定要测试的模型。
可选值: `rnn`, `transformer`, `mt5`, `all` (默认)。

**只测试 MT5:**
```bash
python inference.py --model_type mt5
```

**只测试 Transformer:**
```bash
python inference.py --model_type transformer
```

### 3. 指定解码策略
使用 `--decode` 参数选择解码方式。
可选值: `greedy` (默认), `beam`。

**使用 Beam Search (集束搜索):**
```bash
python inference.py --decode beam --beam_size 4
```

### 4. 数据路径
脚本默认使用以下数据路径：
- RNN/Transformer: `processed_data/test.jsonl` (预处理后的 Token 数据)
- MT5: `data/test.jsonl` (原始文本数据)

如果需要修改，可以使用 `--processed_data` 和 `--raw_data` 参数。

## 模型权重路径
脚本默认从以下路径加载模型：
- RNN: `best_weight/RNN/best.pt`
- Transformer: `best_weight/Transformer/best.pt`
- MT5: `best_weight/MT5/best_model`

如果您的权重文件在其他位置，可以通过 `--rnn_path`, `--transformer_path`, `--mt5_path` 参数指定。


