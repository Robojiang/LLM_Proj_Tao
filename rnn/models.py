import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    支持三种对齐函数：
    - dot:       score(h_t, h_s) = h_t · h_s
    - general:   score(h_t, h_s) = h_t · W_a h_s
    - additive:  score(h_t, h_s) = v_a^T tanh(W_t h_t + W_s h_s)
    """
    def __init__(self, hidden_size, method="dot"):
        super().__init__()
        self.method = method
        if method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "additive":
            self.Wt = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, values, mask):
        # query: [B, H], keys/values: [B, T, H], mask: [B, T] (True for pad)
        if self.method == "dot":
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)          # [B,T]
        elif self.method == "general":
            scores = torch.bmm(self.Wa(keys), query.unsqueeze(-1)).squeeze(-1) # [B,T]
        else:  # additive
            t = self.Wt(query).unsqueeze(1) + self.Ws(keys)                   # [B,T,H]
            scores = self.v(torch.tanh(t)).squeeze(-1)                       # [B,T]
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)                                     # [B,T]
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)                # [B,H]
        return ctx, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=2, pad_idx=0, embeddings=None, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers>1 else 0.0)

    def forward(self, src, src_lens):
        # src: [B,T] T=seq_len 设置的等长序列
        emb = self.embedding(src)                       # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed)              # outputs: Packed, hidden: [L,B,H] L=num_layers，GRU 层数
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # [B,T,H]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, attn_type="dot", num_layers=2, pad_idx=0, embeddings=None, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        self.gru = nn.GRU(emb_dim + hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers>1 else 0.0)
        self.attn = Attention(hidden_size, attn_type)
        self.fc_out = nn.Linear(hidden_size*2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, encoder_outputs, src_mask):
        # input_tok: [B], hidden: [L,B,H], encoder_outputs: [B,T,H], src_mask: [B,T] (pad=True)
        emb = self.dropout(self.embedding(input_tok)).unsqueeze(1) # [B,1,E]
        query = hidden[-1]                                         # [B,H]
        ctx, attn = self.attn(query, encoder_outputs, encoder_outputs, src_mask)  # ctx: [B,H]
        rnn_input = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)     # [B,1,E+H]
        output, hidden = self.gru(rnn_input, hidden)               # output: [B,1,H]
        output = output.squeeze(1)                                 # [B,H]
        logits = self.fc_out(torch.cat([output, ctx], dim=-1))     # [B,V]
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, hidden_size, pad_idx_src, pad_idx_tgt,
                 src_embeddings=None, tgt_embeddings=None, attn_type="dot", num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_size, num_layers, pad_idx_src, src_embeddings, dropout)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_size, attn_type, num_layers, pad_idx_tgt, tgt_embeddings, dropout)
        self.pad_idx_tgt = pad_idx_tgt

    def forward(self, src, src_lens, tgt_inputs, teacher_forcing_ratio=1.0):
        # tgt_inputs: [B,T] 含 <SOS> 开头
        B, T_tgt = tgt_inputs.size()
        device = src.device
        enc_outs, hidden = self.encoder(src, src_lens)
        src_mask = (src == self.encoder.embedding.padding_idx)     # [B,T_src]

        logits_list = []
        inp = tgt_inputs[:,0]  # <SOS>
        for t in range(1, T_tgt):
            logits, hidden, _ = self.decoder(inp, hidden, enc_outs, src_mask)
            logits_list.append(logits.unsqueeze(1))
            teacher = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(-1)
            inp = tgt_inputs[:,t] if teacher else top1

        return torch.cat(logits_list, dim=1)  # [B, T_tgt-1, V]

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len, sos_idx, eos_idx):
        enc_outs, hidden = self.encoder(src, src_lens)
        src_mask = (src == self.encoder.embedding.padding_idx)
        B = src.size(0)
        inp = torch.full((B,), sos_idx, device=src.device, dtype=torch.long)
        outputs = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(inp, hidden, enc_outs, src_mask)
            inp = logits.argmax(-1)
            outputs.append(inp.unsqueeze(1))
            if (inp == eos_idx).all():
                break
        return torch.cat(outputs, dim=1)  # [B, L]

    @torch.no_grad()
    def beam_search(self, src, src_lens, max_len, sos_idx, eos_idx, beam_size=4, len_norm=True):
        # 简洁实现：batch=1 使用 beam
        assert src.size(0) == 1, "beam_search 仅支持单样本解码"
        enc_outs, hidden = self.encoder(src, src_lens)
        src_mask = (src == self.encoder.embedding.padding_idx)

        beams = [(0.0, [sos_idx], hidden)]  # (logprob, seq, hidden)
        for _ in range(max_len):
            new_beams = []
            for logp, seq, h in beams:
                inp = torch.tensor([seq[-1]], device=src.device)
                logits, h_new, _ = self.decoder(inp, h, enc_outs, src_mask)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_logp, topk_idx = log_probs.topk(beam_size)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    new_seq = seq + [idx]
                    new_logp = logp + lp
                    new_beams.append((new_logp, new_seq, h_new))
            beams = sorted(new_beams, key=lambda x: x[0]/(len(x[1]) if len_norm else 1), reverse=True)[:beam_size]
            if all(seq[-1] == eos_idx for _, seq, _ in beams):
                break
        best_seq = beams[0][1][1:]  # 去掉 SOS
        # 截到 EOS
        if eos_idx in best_seq:
            best_seq = best_seq[:best_seq.index(eos_idx)+1]
        return torch.tensor(best_seq, device=src.device).unsqueeze(0)