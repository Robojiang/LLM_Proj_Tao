import os, json, torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import sacrebleu
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help="微调后模型的路径")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--test_file', default='valid.jsonl', help="测试文件名")
    parser.add_argument('--max_len', type=int, default=80, help="生成最大长度")
    parser.add_argument('--num_beams', type=int, default=4, help="Beam search 的 beam 数量")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"加载模型: {args.model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    hyps, refs = [], []
    test_path = os.path.join(args.data_dir, args.test_file)
    print(f"加载测试数据: {test_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="翻译中"):
            pair = json.loads(line)
            src = "translate Chinese to English: " + pair['zh']
            ref = pair['en']

            input_ids = tokenizer(src, return_tensors='pt', max_length=args.max_len, truncation=True).input_ids.to(args.device)
            
            # 使用 beam search 和防重复机制
            outputs = model.generate(
                input_ids, 
                max_length=args.max_len,
                num_beams=args.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,  # 防止重复
                length_penalty=1.0
            )
            hyp = tokenizer.decode(outputs[0], skip_special_tokens=True)

            hyps.append(hyp)
            refs.append([ref])

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"\n{'='*50}")
    print(f"T5 BLEU Score: {bleu.score:.2f}")
    print(f"{'='*50}\n")
    
    print("示例翻译（前 5 条）:")
    for i in range(min(5, len(hyps))):
        print(f"\n--- 样本 {i+1} ---")
        print(f"源: {list(zip(*refs))[0][i]}")  # 获取原始中文（从 refs 反推）
        print(f"预测: {hyps[i]}")
        print(f"参考: {refs[i][0]}")

if __name__ == '__main__':
    main()