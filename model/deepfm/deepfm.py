import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, field_sizes, emb_size, hid_size, dropout=0.7):
        super(DeepFM, self).__init__()

        self.m = len(field_sizes)
        self.f = field_sizes
        self.k = emb_size

        self.order_1_embeddings = nn.ModuleList([nn.Embedding(field_size, 1) for field_size in self.f])
        self.order_2_embeddings = nn.ModuleList([nn.Embedding(field_size, self.k) for field_size in self.f])
        self.dnn = nn.Sequential(nn.Linear(self.k * self.m, hid_size),
                                 nn.Dropout(dropout),
                                 nn.Linear(hid_size, hid_size),
                                 nn.Dropout(dropout),
                                 nn.Linear(hid_size, 1))
        self.out = nn.Sigmoid()

    def forward(self, x):
        """Order 1 connections"""
        order_1_embeddeds = [self.order_1_embeddings[i](x[:, i]) for i in range(self.m)]

        """Order 2 connections"""
        order_2_embeddeds = [self.order_2_embeddings[i](x[:, i]) for i in range(self.m)]

        """FM part"""
        order_1_products = sum(order_1_embeddeds)

        p1 = torch.pow(sum(order_2_embeddeds), 2)
        p2 = sum([torch.pow(e, 2) for e in order_2_embeddeds])
        order_2_products = torch.sum((p1 - p2) / 2, 1).unsqueeze(1)

        y_fm = order_1_products + order_2_products

        """DNN part"""
        y_dnn = self.dnn(torch.cat(order_2_embeddeds, 1))

        return self.out(y_fm + y_dnn)
