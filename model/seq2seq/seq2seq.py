import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    """Seq2Seq with alignment model

    Configs:
        encoder: bidirectional
        decoder: 1 layer
        alignment: feed-forward with no activation
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, tf_rate=0):
        batch_size, trg_len = trg.shape
        decoder_outputs = torch.zeros(trg_len, batch_size, self.decoder.d)
        predictions = torch.zeros(trg_len, batch_size)

        encoder_output, s = self.encoder(src)
        decoder_input = trg[:, 0]
        for i in range(1, trg_len):
            decoder_output, s = self.decoder(decoder_input, s, encoder_output)
            decoder_outputs[i] = decoder_output

            # Top 1 prediction
            prediction = decoder_output.argmax(1)
            predictions[i] = prediction

            decoder_input = trg[i] if random.random() < tf_rate else prediction

        return decoder_outputs, predictions


class Attention(nn.Module):
    """Alignment model"""
    def __init__(self, hid_size):
        super(Attention, self).__init__()

        self.align = nn.Sequential(nn.Linear(hid_size * 3, hid_size), nn.Linear(hid_size, 1))

    def forward(self, s, h):
        src_len = h.shape[1]
        # s = [batch_size, src_len, hid_size]
        s = s.repeat(1, src_len, 1)

        # Alignment model
        # e = Attention(s, h)
        # a = Softmax(e)
        e = self.align(torch.cat((s, h), dim=2))
        # a = [batch_size, src_len]
        a = torch.softmax(e, dim=1).squeeze(2)

        return a


class Encoder(nn.Module):
    """Encoder of the Seq2Seq model"""
    def __init__(self, vocab_size, emb_size, hid_size, n_layers=1, dropout=0.9):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hid_size, n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        encoder_output, hidden = self.rnn(embedded)

        # Extract both of the directions to initialize the hidden state of the decoder
        # encoder_output = [batch_size, src_len, hid_size * 2]
        # hidden = [batch_size, 1, hid_size]
        hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)).unsqueeze(1)

        return encoder_output, hidden


class Decoder(nn.Module):
    """Decoder of the Seq2Seq model"""
    def __init__(self, vocab_size, emb_size, hid_size, attention, dropout=0.9):
        super(Decoder, self).__init__()

        self.d = vocab_size

        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(hid_size * 2 + emb_size, hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input, s, h):
        # decoder_input = [batch_size, 1]
        decoder_input = decoder_input.unsqueeze(1)
        embedded = self.dropout(self.embedding(decoder_input))

        # a = [batch_size, 1, src_len]
        a = self.attention(s, h)
        a = a.unsqueeze(1)

        # c = sigma_j(a_ij, h_j)
        # c = [batch_size, 1, hid_size * 2]
        c = torch.bmm(a, h)

        # decoder_output = [batch_size, 1, hid_size]
        # s = [batch_size, 1, hid_size]
        decoder_output, s = self.rnn(torch.cat((embedded, c), dim=2), s.permute(1, 0, 2))
        # decoder_output = [batch_size, vocab_size]
        decoder_output = self.fc(decoder_output.squeeze(1))

        return decoder_output, s.permute(1, 0, 2)
