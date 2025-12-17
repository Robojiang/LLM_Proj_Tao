import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """实现 RMSNorm，作为 LayerNorm 的替代方案"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class PositionalEncoding(nn.Module):
    """绝对位置编码 (sin/cos)"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RelativeMultiHeadAttention(nn.Module):
    """带相对位置编码的多头注意力机制（Shaw et al., 2018）"""
    def __init__(self, d_model, nhead, dropout=0.1, max_relative_position=32):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_relative_position = max_relative_position
        
        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 相对位置编码的 Key 和 Value
        vocab_size = max_relative_position * 2 + 1
        self.relative_key = nn.Embedding(vocab_size, self.d_k)
        self.relative_value = nn.Embedding(vocab_size, self.d_k)
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_relative_positions(self, length):
        """生成相对位置索引矩阵 [length, length]"""
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # [L, L]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_k, D]
            mask: [T_q, T_k] 因果掩码
            key_padding_mask: [B, T_k] padding 掩码
        """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key.size(1)
        
        # 线性投影并分头: [B, T, D] -> [B, T, H, D_k] -> [B, H, T, D_k]
        Q = self.w_q(query).view(batch_size, len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, len_k, self.nhead, self.d_k).transpose(1, 2)
        
        # 标准注意力分数: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T_q, T_k]
        
        # 添加相对位置编码到注意力分数
        rel_positions = self._get_relative_positions(len_q).to(query.device)  # [T_q, T_k]
        rel_keys = self.relative_key(rel_positions)  # [T_q, T_k, D_k]
        
        # 计算 Q @ relative_K^T 并加到 scores 上
        # Q: [B, H, T_q, D_k], rel_keys: [T_q, T_k, D_k]
        # 需要对每个头单独计算
        rel_scores = torch.einsum('bhqd,qkd->bhqk', Q, rel_keys)  # [B, H, T_q, T_k]
        scores = scores + rel_scores
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == True, float('-inf'))
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T_q, T_k]
        attn_weights = self.dropout(attn_weights)
        
        # 标准的 Attention @ V
        context = torch.matmul(attn_weights, V)  # [B, H, T_q, D_k]
        
        # 添加相对位置编码到 Value
        rel_values = self.relative_value(rel_positions)  # [T_q, T_k, D_k]
        rel_context = torch.einsum('bhqk,qkd->bhqd', attn_weights, rel_values)  # [B, H, T_q, D_k]
        context = context + rel_context
        
        # 合并多头: [B, H, T_q, D_k] -> [B, T_q, H, D_k] -> [B, T_q, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)
        
        # 最终线性投影
        output = self.w_o(context)
        return output

class RelativeTransformerEncoderLayer(nn.Module):
    """使用相对位置编码的 Transformer Encoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_relative_position=32):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout, max_relative_position)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward with residual
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class RelativeTransformerDecoderLayer(nn.Module):
    """使用相对位置编码的 Transformer Decoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_relative_position=32):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout, max_relative_position)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with causal mask
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention (标准的，不用相对位置)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory, 
                                   key_padding_mask=memory_key_padding_mask,
                                   attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, pos_embedding_type='absolute', norm_type='layernorm',
                 src_pad_idx=0, tgt_pad_idx=0, src_embeddings=None, tgt_embeddings=None, 
                 emb_dim=300, max_relative_position=32):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model
        self.pos_embedding_type = pos_embedding_type

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=tgt_pad_idx)
        if src_embeddings is not None:
            self.src_embedding.weight.data.copy_(src_embeddings)
        if tgt_embeddings is not None:
            self.tgt_embedding.weight.data.copy_(tgt_embeddings)

        # 线性投影层
        self.src_proj = nn.Linear(emb_dim, d_model)
        self.tgt_proj = nn.Linear(emb_dim, d_model)

        # 位置编码
        if pos_embedding_type == 'absolute':
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        elif pos_embedding_type == 'relative':
            # 相对位置编码不需要显式的位置编码层
            pass
        else:
            raise ValueError(f"不支持的位置编码类型: {pos_embedding_type}")

        # 标准化层
        if norm_type == 'layernorm':
            encoder_norm = nn.LayerNorm(d_model)
            decoder_norm = nn.LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            encoder_norm = RMSNorm(d_model)
            decoder_norm = RMSNorm(d_model)
        else:
            raise ValueError("标准化类型必须是 'layernorm' 或 'rmsnorm'")

        # Transformer 核心
        if pos_embedding_type == 'absolute':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)
        else:  # relative
            self.encoder_layers = nn.ModuleList([
                RelativeTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, max_relative_position)
                for _ in range(num_encoder_layers)
            ])
            self.encoder_norm = encoder_norm
            
            self.decoder_layers = nn.ModuleList([
                RelativeTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, max_relative_position)
                for _ in range(num_decoder_layers)
            ])
            self.decoder_norm = decoder_norm

        # 输出层
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, src_lens, tgt_inputs, teacher_forcing_ratio=1.0):
        src_key_padding_mask = (src == self.src_pad_idx)
        tgt_key_padding_mask = (tgt_inputs == self.tgt_pad_idx)
        tgt_mask = self.generate_square_subsequent_mask(tgt_inputs.size(1)).to(src.device)

        # 嵌入 + 投影
        src_emb = self.src_proj(self.src_embedding(src)) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_proj(self.tgt_embedding(tgt_inputs)) * math.sqrt(self.d_model)

        # 位置编码
        if self.pos_embedding_type == 'absolute':
            src_emb = self.pos_encoder(src_emb)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            output = self.transformer_decoder(
                tgt_emb, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
        else:  # relative
            # Encoder
            memory = src_emb
            for layer in self.encoder_layers:
                memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
            memory = self.encoder_norm(memory)
            
            # Decoder
            output = tgt_emb
            for layer in self.decoder_layers:
                output = layer(output, memory, tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)
            output = self.decoder_norm(output)
        
        return self.generator(output)

    def greedy_decode(self, src, src_lens, max_len, sos_idx, eos_idx):
        self.eval()
        src_key_padding_mask = (src == self.src_pad_idx)
        src_emb = self.src_proj(self.src_embedding(src)) * math.sqrt(self.d_model)
        
        if self.pos_embedding_type == 'absolute':
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        else:  # relative
            memory = src_emb
            for layer in self.encoder_layers:
                memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
            memory = self.encoder_norm(memory)
        
        ys = torch.ones(1, 1).fill_(sos_idx).type_as(src.data).long()
        
        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            tgt_emb = self.tgt_proj(self.tgt_embedding(ys)) * math.sqrt(self.d_model)
            
            if self.pos_embedding_type == 'absolute':
                tgt_emb = self.pos_encoder(tgt_emb)
                out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            else:  # relative
                out = tgt_emb
                for layer in self.decoder_layers:
                    out = layer(out, memory, tgt_mask=tgt_mask)
                out = self.decoder_norm(out)

            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).long()], dim=1)
            if next_word == eos_idx:
                break
        return ys

    def beam_search(self, src, src_lens, max_len, sos_idx, eos_idx, beam_size=4):
        """完整的 Beam Search 实现"""
        self.eval()
        device = src.device
        src_key_padding_mask = (src == self.src_pad_idx)
        src_emb = self.src_proj(self.src_embedding(src)) * math.sqrt(self.d_model)
        
        # Encode 源句子
        if self.pos_embedding_type == 'absolute':
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        else:
            memory = src_emb
            for layer in self.encoder_layers:
                memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
            memory = self.encoder_norm(memory)
        
        # 初始化 beam：每个 beam 是 (序列, 累积log概率)
        beams = [(torch.ones(1, 1).fill_(sos_idx).type_as(src.data).long(), 0.0)]
        completed_sequences = []
        
        for step in range(max_len - 1):
            all_candidates = []
            
            for seq, score in beams:
                # 如果已经生成 EOS，直接加入完成列表
                if seq[0, -1].item() == eos_idx:
                    completed_sequences.append((seq, score))
                    continue
                
                # 生成当前序列的下一个词的概率分布
                tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                tgt_emb = self.tgt_proj(self.tgt_embedding(seq)) * math.sqrt(self.d_model)
                
                if self.pos_embedding_type == 'absolute':
                    tgt_emb = self.pos_encoder(tgt_emb)
                    out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                else:
                    out = tgt_emb
                    for layer in self.decoder_layers:
                        out = layer(out, memory, tgt_mask=tgt_mask)
                    out = self.decoder_norm(out)
                
                logits = self.generator(out[:, -1])  # [1, vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
                
                # 取 top-k 个候选词
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)
                
                # 扩展当前序列
                for i in range(beam_size):
                    next_word = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # [1, 1]
                    next_seq = torch.cat([seq, next_word], dim=1)
                    next_score = score + topk_log_probs[0, i].item()
                    all_candidates.append((next_seq, next_score))
            
            # 如果所有 beam 都已完成，提前结束
            if not all_candidates:
                break
            
            # 按分数排序，保留 top beam_size 个候选
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]
        
        # 合并未完成和已完成的序列
        all_sequences = completed_sequences + beams
        all_sequences.sort(key=lambda x: x[1] / len(x[0][0]), reverse=True)  # 按平均log概率排序
        
        # 返回最佳序列
        best_seq, best_score = all_sequences[0]
        return best_seq

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask