import os, json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import wandb
from datetime import datetime

# --- PEFT (LoRA) 支持 ---
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class T5NMTDataset(Dataset):
    def __init__(self, jsonl_path, max_len=128):
        """使用原始数据格式"""
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                # 直接使用原始的中英文句子
                src = pair['zh']
                tgt = pair['en']
                self.data.append((src, tgt))
        self.max_len = max_len
        print(f"加载了 {len(self.data)} 条数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'src': self.data[idx][0], 'tgt': self.data[idx][1]}

def collate_fn(batch, tokenizer, max_len):
    """动态填充的 collate 函数"""
    srcs = ["translate Chinese to English: " + item['src'] for item in batch]
    tgts = [item['tgt'] for item in batch]
    
    # 编码源句子和目标句子
    src_enc = tokenizer(srcs, max_length=max_len, padding=True, truncation=True, return_tensors='pt')
    tgt_enc = tokenizer(tgts, max_length=max_len, padding=True, truncation=True, return_tensors='pt')
    
    # 将目标的 padding token 替换为 -100（忽略 loss 计算）
    labels = tgt_enc['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': src_enc['input_ids'],
        'attention_mask': src_enc['attention_mask'],
        'labels': labels
    }

@hydra.main(config_path="config", config_name="finetune_raw", version_base=None)
def main(cfg: DictConfig):
    # --- 路径和设备 ---
    data_dir = to_absolute_path(cfg.data.dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = cfg.model.name.replace('/', '-')
    output_dir = os.path.join("./runs", f"{timestamp}_{cfg.experiment}_{model_name_safe}_{cfg.model.finetune_method}_raw")
    os.makedirs(output_dir, exist_ok=True)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # --- W&B 初始化 ---
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name or os.path.basename(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # --- 加载模型和分词器 ---
    print(f"加载模型: {cfg.model.name}")
    tokenizer = MT5Tokenizer.from_pretrained(cfg.model.name)
    model = MT5ForConditionalGeneration.from_pretrained(cfg.model.name,use_safetensors=True)

    # --- 微调方法选择 ---
    if cfg.model.finetune_method == 'lora':
        if not PEFT_AVAILABLE:
            raise ImportError("请安装 peft 库: pip install peft")
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
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数量: {num_params:,}")

    model.to(device)

    # --- 数据加载 ---
    print(f"加载训练数据: {cfg.data.train_file}")
    train_ds = T5NMTDataset(os.path.join(data_dir, cfg.data.train_file), cfg.data.max_len)
    print(f"加载验证数据: {cfg.data.valid_file}")
    valid_ds = T5NMTDataset(os.path.join(data_dir, cfg.data.valid_file), cfg.data.max_len)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, cfg.data.max_len)
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=cfg.train.batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, cfg.data.max_len)
    )

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
    best_val_loss = float('inf')
    if cfg.train.resume_from:
        checkpoint = torch.load(to_absolute_path(cfg.train.resume_from), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"🔄 恢复训练，从 epoch {start_epoch} 开始，最佳验证损失: {best_val_loss:.4f}")

    # --- 训练循环 ---
    print(f"\n开始训练，共 {cfg.train.epochs} 个 epoch，每 {cfg.train.save_every} 个 epoch 保存一次检查点\n")
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            
            if cfg.wandb.enable:
                wandb.log({"step_loss": loss.item(), "lr": current_lr})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - 平均训练损失: {avg_train_loss:.4f}")

        # --- 验证 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="验证中"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1} - 验证损失: {avg_val_loss:.4f}\n")
        
        if cfg.wandb.enable:
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"✅ 最佳模型已保存到: {best_model_path} (验证损失: {avg_val_loss:.4f})\n")

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
            print(f"✅ 检查点已保存: {checkpoint_path}\n")

    # --- 保存最终模型 ---
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ 最终模型已保存到: {final_model_path}")
    if cfg.wandb.enable:
        wandb.finish()

if __name__ == '__main__':
    main()