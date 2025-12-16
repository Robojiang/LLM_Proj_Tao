import os, json, pickle, argparse
import torch
from torch.utils.data import Dataset, DataLoader
import sacrebleu
from rnn.models import Seq2Seq
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--test_file', default='test.jsonl')
    parser.add_argument('--ckpt', default='./runs/rnn-gru_rnn_additive_20251216_170754/epoch_20.pt')
    parser.add_argument('--decode', choices=['greedy','beam'], default='greedy')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ckpt['config']
    zh_vocab = ckpt['zh_vocab']; en_vocab = ckpt['en_vocab']
    id2en = {v:k for k,v in en_vocab.items()}
    pad_src, pad_tgt = zh_vocab[PAD], en_vocab[PAD]

    test_ds = NMTDataset(os.path.join(args.data_dir, args.test_file), zh_vocab, args.max_len)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_src))

    model = Seq2Seq(len(zh_vocab), len(en_vocab), cfg['model']['emb_dim'], cfg['model']['hidden'], pad_src, pad_tgt,
                    src_embeddings=None, tgt_embeddings=None, attn_type=cfg['model']['attn'], num_layers=2).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    hyps, refs = [], []
    with torch.no_grad():
        # 包装 loader，显示总步数
        pbar = tqdm(loader, desc="推理中", total=len(loader))
        for src, src_lens, tgt_ref in pbar:
            src, src_lens = src.to(args.device), src_lens.to(args.device)
            if args.decode == 'greedy':
                out_ids = model.greedy_decode(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS])
            else:
                out_ids = model.beam_search(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS], beam_size=args.beam_size)
            toks = ids_to_tokens(out_ids[0].tolist(), id2en)
            hyps.append(detokenize(toks))
            if tgt_ref[0] is not None:
                refs.append([detokenize(tgt_ref[0])])
    
    if refs:
        bleu = sacrebleu.corpus_bleu(hyps, refs)
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