import torch
import torch.nn as nn
from .transformer import EncoderLayer


class BERT(nn.Module):
    def __init__(self, input_size, hid_size, n_heads=8, n_layers=12, dropout=0.9, max_len=128):
        super(BERT, self).__init__()
        self.hid_size = hid_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.tok_embedding = nn.Embedding(input_size, hid_size)
        self.pos_embedding = nn.Embedding(max_len, hid_size)
        self.seg_embedding = nn.Embedding(2, hid_size)
        self.encoders = nn.ModuleList([EncoderLayer(hid_size, n_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, segment_ids):
        batch_size, max_len = x.shape
        mask = (x > 0).unsqueeze(1).repeat(1, max_len, 1).unsqueeze(1)

        te = self.tok_embedding(x)
        pos = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1)
        pe = self.pos_embedding(pos)
        se = self.seg_embedding(segment_ids)
        x = te + pe + se
        x = self.dropout(x)

        for encoder in self.encoders:
            x = encoder(x, mask)

        return x


class PreTrainModel(nn.Module):
    def __init__(self, bert):
        super(PreTrainModel, self).__init__()

        self.mlm = MaskedLanguageModel(bert.vocab_size, bert.hid_size)
        self.nsp = NextSentencePrediction(bert.hid_size)

    def forward(self, x, segment_ids):
        x = self.bert(x, segment_ids)

        prediction_scores = self.mlm(x)
        next_sentence_scores = self.nsp(x)

        return prediction_scores, next_sentence_scores


class NextSentencePrediction(nn.Module):
    def __init__(self, hid_size):
        super(NextSentencePrediction, self).__init__()

        self.fc = nn.Linear(hid_size, 2)

    def forward(self, x):
        return torch.softmax(self.fc(x[:, 0, :]), dim=-1)


class MaskedLanguageModel(nn.Module):
    def __init__(self, output_size, hid_size):
        super(MaskedLanguageModel, self).__init__()

        self.fc = nn.Linear(hid_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
