import os, json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

class T5NMTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                src = "translate Chinese to English: " + "".join(pair['zh_tokens'])
                tgt = "".join(pair['en_tokens'])
                self.data.append((src, tgt))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_enc = self.tokenizer(src, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': tgt_enc['input_ids'].squeeze()
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='t5-small')
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--train_file', default='train.jsonl')
    parser.add_argument('--valid_file', default='valid.jsonl')
    parser.add_argument('--output_dir', default='./runs/t5_finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)

    # 数据加载
    train_ds = T5NMTDataset(os.path.join(args.data_dir, args.train_file), tokenizer)
    valid_ds = T5NMTDataset(os.path.join(args.data_dir, args.valid_file), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1} - Valid Loss: {val_loss:.4f}")

    # 保存模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到 {args.output_dir}")

if __name__ == '__main__':
    main()