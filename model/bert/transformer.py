import torch
import torch.nn as nn
import numpy as np


class EncoderLayer(nn.Module):
    def __init__(self, hid_size, n_heads, dropout=0.9):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(hid_size, n_heads)
        self.ffn = nn.Sequential(nn.Linear(hid_size, hid_size * 4), nn.GELU(), nn.Linear(hid_size * 4, hid_size))
        self.layer_norm = nn.LayerNorm(hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        attention_output = self.layer_norm(x + self.dropout(attention_output))

        encoder_output = self.ffn(attention_output)
        encoder_output = self.layer_norm(attention_output + self.dropout(encoder_output))

        return encoder_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""

    def __init__(self, hid_size, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.h = n_heads
        self.d_k = hid_size // n_heads

        self.w_q = nn.Linear(hid_size, hid_size)
        self.w_k = nn.Linear(hid_size, hid_size)
        self.w_v = nn.Linear(hid_size, hid_size)
        self.w_o = nn.Linear(hid_size, hid_size)

    def forward(self, query, key, value, mask=None):
        # q, k, v = [batch_size, src_len, hid_size]
        batch_size, hid_size = query.shape[0], query.shape[2]

        # q, k, v = [batch_size, src_len, hid_size]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # q, v = [batch_size, src_len, n_heads, head_size]
        # k = [batch_size, src_len, head_size, n_heads]
        q = q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 3, 1)
        v = v.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)

        # Attention(Q, K, V) = Softmax(Q * K^T / d) * V
        attention_scores = torch.matmul(q, k) / np.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        attention = torch.softmax(attention_scores, dim=-1)
        y = torch.matmul(attention, v)

        # y = [batch_size, src_len, hid_size]
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hid_size)

        return self.w_o(y)
