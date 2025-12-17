import os, json, torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import sacrebleu
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./runs/t5_finetuned')
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--test_file', default='valid.jsonl')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    hyps, refs = [], []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="翻译中"):
            pair = json.loads(line)
            src = "translate Chinese to English: " + "".join(pair['zh_tokens'])
            ref = "".join(pair['en_tokens'])

            input_ids = tokenizer(src, return_tensors='pt').input_ids.to(args.device)
            outputs = model.generate(input_ids, max_length=128)
            hyp = tokenizer.decode(outputs[0], skip_special_tokens=True)

            hyps.append(hyp)
            refs.append([ref])

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"T5 BLEU Score: {bleu.score:.2f}")

if __name__ == '__main__':
    main()