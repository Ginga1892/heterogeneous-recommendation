import torch
import torch.nn as nn

# class Transformer(nn.Module):
#     """Transformer"""
#
#     def __init__(self):
#         super(Transformer, self).__init__()


class Encoder(nn.Module):
    """Encoder of the Transformer model

    Simplified:
        layers: 6
        heads: 8
        layers of feed-forward networks: 1
    """

    def __init__(self, input_size, hid_size, dropout=0.7, max_len=100):
        super(Encoder, self).__init__()

        self.n = 6
        self.h = 8
        self.d = torch.sqrt(torch.FloatTensor([hid_size]))

        self.tok_embedding = nn.Embedding(input_size, hid_size)
        self.pos_embedding = nn.Embedding(max_len, hid_size)
        self.self_attention = MultiHeadAttention(hid_size, self.h)
        self.ffn = nn.Linear(hid_size, hid_size)
        self.layer_norm = nn.LayerNorm(hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        batch_size, src_len = src.shape

        # Token embeddings
        te = self.tok_embedding(src) * self.d
        # Positional embeddings
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
        pe = self.pos_embedding(pos)
        src = te + pe
        src = self.dropout(src)

        # Stacked transformer
        # multi-head attention layer
        # add & layer normalization
        # feed-forward networks
        # add & layer normalization
        for i in range(self.n):
            attention_output = self.self_attention(src, src, src, src_mask)
            attention_output = self.layer_norm(src + self.dropout(attention_output))

            encoder_output = self.ffn(attention_output)
            encoder_output = self.layer_norm(attention_output + self.dropout(encoder_output))

        return encoder_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""

    def __init__(self, hid_size, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.h = n_heads

        self.w_q = nn.Linear(hid_size, hid_size)
        self.w_k = nn.Linear(hid_size, hid_size)
        self.w_v = nn.Linear(hid_size, hid_size)
        self.w_o = nn.Linear(hid_size, hid_size)
        self.d = torch.sqrt(torch.FloatTensor([hid_size // self.h]))

    def forward(self, query, key, value, mask=None):
        # q, k, v = [batch_size, src_len]
        batch_size, hid_size = query.shape[0], query.shape[2]

        # q, k, v = [batch_size, src_len, hid_size]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # q, v = [batch_size, src_len, n_heads, head_size]
        # k = q, k = [batch_size, src_len, head_size, n_heads]
        q = q.view(batch_size, -1, self.h, hid_size // self.h).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.h, hid_size // self.h).permute(0, 2, 3, 1)
        v = v.view(batch_size, -1, self.h, hid_size // self.h).permute(0, 2, 1, 3)

        # Attention(Q, K, V) = softmax(Q * K^T / d) * V
        attention_scores = torch.matmul(q, k) / self.d
        if mask:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        attention = torch.softmax(attention_scores, dim=-1)
        y = torch.matmul(attention, v)

        # y = [batch_size, src_len, hid_size]
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hid_size)

        return self.w_o(y)
