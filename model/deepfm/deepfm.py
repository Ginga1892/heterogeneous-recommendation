import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """DeepFM

    Simplified:
        DNN layers: 3
        feature type: discrete
    """

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
        """
        Args:
            x: a tensor of [batch_size, m], tensor([[2, 0, 1],
                                                    [1, 3, 0]])
        """

        # Order 1 connections
        # order_1_embedded = [batch_size, 1] * m
        order_1_embedded = [self.order_1_embeddings[i](x[:, i]) for i in range(self.m)]

        # Order 2 connections
        # order_1_embedded = [batch_size, emb_size] * m
        order_2_embedded = [self.order_2_embeddings[i](x[:, i]) for i in range(self.m)]

        # FM part
        order_1_product = sum(order_1_embedded)

        p1 = torch.pow(sum(order_2_embedded), 2)
        p2 = sum([torch.pow(e, 2) for e in order_2_embedded])
        order_2_product = torch.sum((p1 - p2) / 2, 1).unsqueeze(1)

        # y_fm = [batch_size, 1]
        y_fm = order_1_product + order_2_product

        # DNN part
        # y_dnn = [batch_size, 1]
        y_dnn = self.dnn(torch.cat(order_2_embedded, 1))

        return self.out(y_fm + y_dnn)
