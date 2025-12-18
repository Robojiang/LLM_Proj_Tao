import os, json, torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from tqdm import tqdm
import sacrebleu
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    parser = argparse.ArgumentParser()
    # --- 修改：参数名更通用 ---
    parser.add_argument('--model_path', default='./runs/20251218_121749_mt5-finetune-raw_google-mt5-small_lora_raw/best_model', help="模型文件夹路径 或 .pt 检查点文件路径")
    # --- 新增：当评估 .pt 文件时，需要指定基础模型 ---
    parser.add_argument('--base_model_name', default='mt5-small', help="基础模型名称 (例如 t5-small)，仅在评估检查点时需要")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--test_file', default='trian_test.jsonl', help="测试文件名")
    parser.add_argument('--max_len', type=int, default=80, help="生成最大长度")
    parser.add_argument('--num_beams', type=int, default=4, help="Beam search 的 beam 数量")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # --- 核心修改：智能加载模型 ---
    if os.path.isdir(args.model_path):
        # 情况 1: 输入是文件夹 (例如 best_model)
        print(f"加载已保存的模型文件夹: {args.model_path}")
        tokenizer = MT5Tokenizer.from_pretrained(args.model_path,local_files_only=True)
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path,use_safetensors=True, local_files_only=True).to(args.device)
    elif args.model_path.endswith('.pt'):
        # 情况 2: 输入是 .pt 检查点文件
        print(f"加载检查点文件: {args.model_path}")
        print(f"使用基础模型 '{args.base_model_name}' 构建结构")
        
        # 先加载基础模型结构和分词器
        tokenizer = MT5Tokenizer.from_pretrained(args.base_model_name)
        model = MT5ForConditionalGeneration.from_pretrained(args.base_model_name)
        
        # 加载检查点中的权重
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
    else:
        raise ValueError("无效的 --model_path，必须是文件夹或 .pt 文件")

    model.eval()

    hyps, refs, sources = [], [], []
    test_path = os.path.join(args.data_dir, args.test_file)
    print(f"加载测试数据: {test_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="翻译中"):
            pair = json.loads(line)
            source_text = pair['zh']
            ref_text = pair['en']
            
            prompted_text = "translate Chinese to English: " + source_text

            input_ids = tokenizer(prompted_text, return_tensors='pt', max_length=args.max_len, truncation=True).input_ids.to(args.device)
            
            outputs = model.generate(
                input_ids, 
                max_length=args.max_len,
                num_beams=args.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            hyp = tokenizer.decode(outputs[0], skip_special_tokens=True)

            hyps.append(hyp)
            sources.append(source_text)
            refs.append([ref_text])

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"\n{'='*50}")
    print(f"T5 BLEU Score: {bleu.score:.2f}")
    print(f"{'='*50}\n")
    
    print("示例翻译（前 5 条）:")
    for i in range(min(5, len(hyps))):
        print(f"\n--- 样本 {i+1} ---")
        print(f"源: {sources[i]}")
        print(f"预测: {hyps[i]}")
        print(f"参考: {refs[i][0]}")

if __name__ == '__main__':
    main()