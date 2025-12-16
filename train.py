import os, json, pickle, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
import wandb
from rnn.models import Seq2Seq  

PAD, UNK, SOS, EOS = '<PAD>', '<UNK>', '<SOS>', '<EOS>'

class NMTDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab, max_len):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                src_toks = pair['zh_tokens']
                tgt_toks = pair['en_tokens']
                if 0 < len(src_toks) <= max_len and 0 < len(tgt_toks) <= max_len:
                    self.data.append((src_toks, tgt_toks))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        src_toks, tgt_toks = self.data[idx]
        src_ids = [self.src_vocab.get(w, self.src_vocab[UNK]) for w in src_toks] + [self.src_vocab[EOS]]
        tgt_ids = [self.tgt_vocab[SOS]] + [self.tgt_vocab.get(w, self.tgt_vocab[UNK]) for w in tgt_toks] + [self.tgt_vocab[EOS]]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_embeddings(npy_path):
    if npy_path and os.path.exists(npy_path):
        return torch.tensor(np.load(npy_path), dtype=torch.float)
    return None

def save_checkpoint(model, optimizer, epoch, best_val, cfg, zh_vocab, en_vocab, path, wandb_id=None):
    """保存完整的 checkpoint"""
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_val': best_val,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'zh_vocab': zh_vocab,
        'en_vocab': en_vocab,
        'wandb_id': wandb_id
    }, path)

def load_checkpoint(path, model, optimizer=None):
    """加载 checkpoint"""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt

def create_exp_dir(cfg):
    """创建实验目录：./runs/{experiment}_{timestamp}"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment}_{cfg.model.attn}_{cfg.model.teacher_forcing}_{cfg.model.hidden}_{timestamp}"
    exp_dir = os.path.join("./runs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置文件
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    return exp_dir

@hydra.main(config_path="config", config_name="rnn", version_base=None)
def main(cfg: DictConfig):
    # 绝对路径
    data_dir = to_absolute_path(cfg.data.dir)
    train_path = os.path.join(data_dir, cfg.data.train_file)
    valid_path = os.path.join(data_dir, cfg.data.valid_file)
    src_emb_path = to_absolute_path(cfg.model.src_emb)
    tgt_emb_path = to_absolute_path(cfg.model.tgt_emb)

    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    set_seed(cfg.seed)

    # 恢复训练 or 新建实验目录
    resume_path = cfg.train.get('resume', None)
    if resume_path and os.path.exists(to_absolute_path(resume_path)):
        # 恢复模式：从 checkpoint 所在目录继续
        exp_dir = os.path.dirname(to_absolute_path(resume_path))
        print(f"🔄 恢复训练从: {resume_path}")
    else:
        # 新训练：创建独立实验目录
        exp_dir = to_absolute_path(create_exp_dir(cfg))
        print(f"📁 创建新实验目录: {exp_dir}")
    
    best_ckpt_path = os.path.join(exp_dir, "best.pt")
    last_ckpt_path = os.path.join(exp_dir, "last.pt")

    # 载入词表
    with open(os.path.join(data_dir, 'zh_vocab.pkl'), 'rb') as f: zh_vocab = pickle.load(f)
    with open(os.path.join(data_dir, 'en_vocab.pkl'), 'rb') as f: en_vocab = pickle.load(f)
    pad_src, pad_tgt = zh_vocab[PAD], en_vocab[PAD]

    # 数据
    train_ds = NMTDataset(train_path, zh_vocab, en_vocab, cfg.data.max_len)
    valid_ds = NMTDataset(valid_path, zh_vocab, en_vocab, cfg.data.max_len)
    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
        tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)
        src_pad = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_src)
        tgt_pad = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_tgt)
        return src_pad, src_lens, tgt_pad, tgt_lens

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    # 模型
    if cfg.model.type != "rnn":
        raise NotImplementedError("Transformer 请后续补充。")
    src_emb = load_embeddings(src_emb_path)
    tgt_emb = load_embeddings(tgt_emb_path)
    model = Seq2Seq(
        len(zh_vocab), len(en_vocab),
        cfg.model.emb_dim, cfg.model.hidden,
        pad_src, pad_tgt,
        src_embeddings=src_emb,
        tgt_embeddings=tgt_emb,
        attn_type=cfg.model.attn,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.get('dropout', 0.1)
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # 尝试恢复训练
    start_epoch = 1
    best_val = 1e9
    wandb_id = None
    if resume_path and os.path.exists(to_absolute_path(resume_path)):
        ckpt = load_checkpoint(to_absolute_path(resume_path), model, optimizer)
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', 1e9)
        wandb_id = ckpt.get('wandb_id', None)
        print(f"✓ 从 epoch {start_epoch} 继续训练，best_val={best_val:.4f}")
    elif os.path.exists(last_ckpt_path):
        print(f"🔍 检测到上次训练的 checkpoint: {last_ckpt_path}")
        ckpt = load_checkpoint(last_ckpt_path, model, optimizer)
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', 1e9)
        wandb_id = ckpt.get('wandb_id', None)
        print(f"✓ 从 epoch {start_epoch} 继续训练，best_val={best_val:.4f}")

    # W&B（恢复或新建）
    if cfg.wandb.enable:
        run_name = cfg.wandb.run_name if cfg.wandb.run_name else os.path.basename(exp_dir)
        if wandb_id:
            wandb.init(project=cfg.wandb.project, name=run_name, id=wandb_id, resume="must",
                       config=OmegaConf.to_container(cfg, resolve=True))
        else:
            wandb.init(project=cfg.wandb.project, name=run_name,
                       config=OmegaConf.to_container(cfg, resolve=True))
            wandb_id = wandb.run.id
        wandb.watch(model, log="all", log_freq=100)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for src, src_lens, tgt, _ in pbar:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            logits = model(src, src_lens, tgt, teacher_forcing_ratio=cfg.model.teacher_forcing)
            tgt_y = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)

        # 验证（free running）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, src_lens, tgt, _ in valid_loader:
                src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
                logits = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
                tgt_y = tgt[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
                val_loss += loss.item()
        val_loss /= len(valid_loader)

        if cfg.wandb.enable:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "teacher_forcing": cfg.model.teacher_forcing
            })

        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} valid_loss={val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, cfg, zh_vocab, en_vocab, best_ckpt_path, wandb_id)
            print(f"✓ 保存最佳模型到 {best_ckpt_path}")

        # 每 5 个 epoch 保存一次
        if epoch % 5 == 0:
            periodic_path = os.path.join(exp_dir, f"epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, best_val, cfg, zh_vocab, en_vocab, periodic_path, wandb_id)
            print(f"✓ 定期保存到 {periodic_path}")

        # 始终保存 last checkpoint（用于断点恢复）
        save_checkpoint(model, optimizer, epoch, best_val, cfg, zh_vocab, en_vocab, last_ckpt_path, wandb_id)

if __name__ == '__main__':
    main()