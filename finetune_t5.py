import os, json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import wandb
from datetime import datetime

# --- 新增：PEFT (LoRA) 支持 ---
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class T5NMTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=50):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                # T5 需要一个任务前缀
                src = "translate Chinese to English: " + "".join(pair['zh_tokens'])
                tgt = " ".join(pair['en_tokens'])
                self.data.append((src, tgt))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_enc = self.tokenizer(src, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        # T5 的 labels 需要是 input_ids
        labels = tgt_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100 # 忽略 padding 部分的 loss

        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': labels
        }

@hydra.main(config_path="config", config_name="finetune", version_base=None)
def main(cfg: DictConfig):
    # --- 路径和设备 ---
    data_dir = to_absolute_path(cfg.data.dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./runs", f"_{timestamp}_{cfg.experiment}_{cfg.model.name}_{cfg.model.finetune_method}")
    os.makedirs(output_dir, exist_ok=True)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # --- W&B 初始化 ---
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name or os.path.basename(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # --- 加载模型和分词器 ---
    tokenizer = T5Tokenizer.from_pretrained(cfg.model.name)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model.name)

    # --- 微调方法选择 ---
    if cfg.model.finetune_method == 'lora':
        print("🚀 使用 LoRA 微调...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            target_modules=list(cfg.model.lora.target_modules)
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("🚀 使用全量微调...")

    model.to(device)

    # --- 数据加载 ---
    train_ds = T5NMTDataset(os.path.join(data_dir, cfg.data.train_file), tokenizer, cfg.data.max_len)
    valid_ds = T5NMTDataset(os.path.join(data_dir, cfg.data.valid_file), tokenizer, cfg.data.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.train.batch_size)

    # --- 优化器和调度器 ---
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    total_steps = len(train_loader) * cfg.train.epochs
    if cfg.train.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.train.warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = None

    # --- 恢复训练 ---
    start_epoch = 0
    best_val_loss = float('inf')  # 初始化最佳验证损失
    if cfg.train.resume_from:
        checkpoint = torch.load(to_absolute_path(cfg.train.resume_from), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"🔄 恢复训练，从 epoch {start_epoch} 开始...")

    # --- 训练循环 ---
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            if cfg.wandb.enable:
                wandb.log({"step_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - 平均训练损失: {avg_train_loss:.4f}")

        # --- 验证 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1} - 验证损失: {avg_val_loss:.4f}")
        
        if cfg.wandb.enable:
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"✅ 最佳模型已保存到: {best_model_path}")

        # --- 保存 checkpoint ---
        if (epoch + 1) % cfg.train.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"✅ 模型检查点已保存到: {checkpoint_path}")

    # --- 保存最终模型 ---
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ 最终模型已保存到: {final_model_path}")
    if cfg.wandb.enable:
        wandb.finish()

if __name__ == '__main__':
    main()