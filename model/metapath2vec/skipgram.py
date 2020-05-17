import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SkipGram, self).__init__()

        self.emb_size = emb_size
        self.u_embedding = nn.Embedding(vocab_size, emb_size)
        self.v_embedding = nn.Embedding(vocab_size, emb_size)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, u, v, neg_v):
        u_embedded = self.u_embedding(u)
        v_embedded = self.v_embedding(v)
        neg_v_embedded = self.v_embedding(neg_v)

        score = torch.sum(torch.mul(u_embedded, v_embedded), dim=1)
        logits = self.sigmoid(score).squeeze()

        neg_v_embedded = self.v_embedding(neg_v)
        neg_score = torch.sum(torch.bmm(neg_v_embedded, u_embedded.unsqueeze(2)), dim=1)
        neg_logits = self.sigmoid(-neg_score).squeeze(1)

        return -(logits + neg_logits).mean()


class HeterogeneousSkipGram(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(HeterogeneousSkipGram, self).__init__()

        self.emb_size = emb_size
        self.u_embedding = nn.Embedding(vocab_size, emb_size, sparse=True)
        self.v_embedding = nn.Embedding(vocab_size, emb_size, sparse=True)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, u, v, neg_v):
        u_embedded = self.u_embedding(u)
        v_embedded = self.v_embedding(v)
        neg_v_embedded = self.v_embedding(neg_v)

        score = torch.sum(torch.mul(u_embedded, v_embedded), dim=1)
        score = torch.clamp(score, min=-10, max=10)
        logits = self.sigmoid(score).squeeze()

        neg_score = torch.sum(torch.bmm(neg_v_embedded, u_embedded.unsqueeze(2)), dim=1)
        neg_score = torch.clamp(neg_score, min=-10, max=10)
        neg_logits = self.sigmoid(-neg_score).squeeze(1)

        return -(logits + neg_logits).mean()
