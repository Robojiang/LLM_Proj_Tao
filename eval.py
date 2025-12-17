import os, json, pickle, argparse
import torch
from torch.utils.data import Dataset, DataLoader
import sacrebleu
from rnn.models import Seq2Seq
from transformer.models import TransformerNMT
from tqdm import tqdm 

PAD, UNK, SOS, EOS = '<PAD>', '<UNK>', '<SOS>', '<EOS>'

class NMTDataset(Dataset):
    def __init__(self, path, src_vocab, max_len):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                src_toks = pair['zh_tokens']
                tgt_toks = pair.get('en_tokens')
                if 0 < len(src_toks) <= max_len:
                    self.data.append((src_toks, tgt_toks))
        self.src_vocab = src_vocab
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        src_toks, tgt_toks = self.data[idx]
        src_ids = [self.src_vocab.get(w, self.src_vocab[UNK]) for w in src_toks] + [self.src_vocab[EOS]]
        return torch.tensor(src_ids), tgt_toks

def collate_fn(batch, pad_idx):
    src_seqs, tgts = zip(*batch)
    lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    return src_pad, lens, tgts

def ids_to_tokens(ids, id2word):
    toks = []
    for i in ids:
        w = id2word.get(int(i), '<UNK>')
        if w == EOS:
            break
        toks.append(w)
    return toks

def detokenize(tokens):
    """
    将 token 列表转为正常句子（仅处理标点符号）
    """
    if isinstance(tokens, list):
        sentence = " ".join(tokens)
    else:
        sentence = tokens
    # 只去掉标点符号前的空格
    sentence = sentence.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
    sentence = sentence.replace(" '", "'").replace(" n't", "n't")
    return sentence.strip()

def load_model(cfg, zh_vocab, en_vocab, pad_src, pad_tgt, device):
    """
    根据配置动态加载模型（支持 RNN 和 Transformer）
    """
    if cfg['model']['type'] == 'rnn':
        model = Seq2Seq(
            len(zh_vocab), len(en_vocab),
            cfg['model']['emb_dim'], cfg['model']['hidden'],
            pad_src, pad_tgt,
            src_embeddings=None, tgt_embeddings=None,
            attn_type=cfg['model']['attn'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model'].get('dropout', 0.1)
        ).to(device)
    elif cfg['model']['type'] == 'transformer':
        model = TransformerNMT(
            src_vocab_size=len(zh_vocab),
            tgt_vocab_size=len(en_vocab),
            d_model=cfg['model']['d_model'],
            nhead=cfg['model']['nhead'],
            num_encoder_layers=cfg['model']['num_encoder_layers'],
            num_decoder_layers=cfg['model']['num_decoder_layers'],
            dim_feedforward=cfg['model']['dim_feedforward'],
            dropout=cfg['model']['dropout'],
            pos_embedding_type=cfg['model']['pos_embedding_type'],
            norm_type=cfg['model']['norm_type'],
            src_pad_idx=pad_src,
            tgt_pad_idx=pad_tgt,
            src_embeddings=None,
            tgt_embeddings=None,
            emb_dim=cfg['model']['emb_dim']
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {cfg['model']['type']}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--test_file', default='test.jsonl')
    parser.add_argument('--ckpt', default='./runs/20251217_140918_transformer_relative_layernorm/epoch_10.pt')
    parser.add_argument('--decode', choices=['greedy','beam'], default='greedy')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ckpt['config']
    zh_vocab = ckpt['zh_vocab']; en_vocab = ckpt['en_vocab']
    id2en = {v:k for k,v in en_vocab.items()}
    pad_src, pad_tgt = zh_vocab[PAD], en_vocab[PAD]

    # 加载测试数据
    test_ds = NMTDataset(os.path.join(args.data_dir, args.test_file), zh_vocab, args.max_len)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_src))

    # 动态加载模型
    model = load_model(cfg, zh_vocab, en_vocab, pad_src, pad_tgt, args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # 推理
    hyps, refs = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="推理中", total=len(loader))
        for src, src_lens, tgt_ref in pbar:
            src, src_lens = src.to(args.device), src_lens.to(args.device)
            if args.decode == 'greedy':
                out_ids = model.greedy_decode(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS])
            else:
                out_ids = model.beam_search(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS], beam_size=args.beam_size)
            toks = ids_to_tokens(out_ids[0].tolist(), id2en)
            toks = toks[1:] if toks[0] == SOS else toks  # 去掉开头的<SOS>
            hyps.append(detokenize(toks))
            if tgt_ref[0] is not None:
                refs.append([detokenize(tgt_ref[0])])
    
    # 计算 BLEU 分数
    if refs:
        bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
        print(f"BLEU = {bleu.score:.2f}")
        print(f"示例翻译（前5条）：")
        for i in range(min(5, len(hyps))):
            print(f"  Hyp: {hyps[i]}")
            print(f"  Ref: {refs[i][0]}")
            print()
    else:
        print("无参考翻译，跳过 BLEU 计算。")

if __name__ == '__main__':
    main()