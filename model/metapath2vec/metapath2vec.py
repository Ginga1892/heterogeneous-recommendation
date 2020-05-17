import torch
import numpy as np
import dgl
from skipgram import HeterogeneousSkipGram


class HN:
    def __init__(self, graph=None, ho=False):
        self.t = 'homo' if ho else 'hetero'
        self.hn = graph if graph else dgl.DGLGraph()

    @classmethod
    def from_edge(cls, edgelist, ho=False):
        hn = cls(ho=ho)
        print(hn.t)
        if hn.t == 'hetero':
            assert isinstance(edgelist, dict)
            hn.hn = dgl.heterograph(edgelist)

        else:
            assert isinstance(edgelist, list)
            hn.hn = dgl.graph(edgelist)

        return hn

    def get_vocab(self):
        self.node2id = {}
        guid = 0

        if self.t == 'hetero':
            for ntype in self.hn.ntypes:
                for node in self.hn.nodes(ntype):
                    node = ntype + str(node.item())
                    self.node2id[node] = guid
                    guid += 1

        else:
            for node in self.hn.nodes():
                node = node.item()
                self.node2id[node] = guid
                guid += 1


class MetaPathRandomWalk:
    def __init__(self, hn, walk_len, metapath):
        self.g = hn
        self.l = walk_len
        self.mp = metapath

    def get_walks(self):
        def walk(start):
            walk = [self.mp[0] + str(start)]
            cur_step = start
            r, n = 1, 2
            while len(walk) < self.l:
                try:
                    next_step = np.random.choice(self.g.hn.successors(cur_step, self.mp[r]))
                except ValueError:
                    break
                walk.append(self.mp[n] + str(next_step))
                cur_step = next_step
                r += 2
                n += 2
            return walk

        walks = []
        for node in self.g.hn.nodes(self.mp[0]):
            walks.append(walk(node.item()))

        return walks


class MetaPath2Vec:
    def __init__(self, hn, metapath, walks_per_node, walk_len, window_size, emb_size, neg_size, batch_size,
                 learning_rate=1e-3):
        self.g = hn
        self.mp = metapath
        self.w = walks_per_node
        self.l = walk_len
        self.d = emb_size
        self.k = window_size
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.random_walk = MetaPathRandomWalk(self.g, self.l, self.mp)
        self.skip_gram = HeterogeneousSkipGram(len(self.g.node2id), self.d)
        self.optimizer = torch.optim.SparseAdam(self.skip_gram.parameters(), lr=learning_rate)

    @staticmethod
    def get_iterator(walks, node2id, batch_size, window_size):
        pairs = []
        for walk in walks:
            for i, u in enumerate(walk):
                for v in walk[max(i - window_size, 0):min(i + window_size + 1, len(walk))]:
                    if u == v:
                        continue
                    pairs.append((node2id[u], node2id[v]))

        iterator = []
        batch, b = [], 0
        while pairs:
            pair = pairs.pop()
            batch.append(pair)
            b += 1

            if b == batch_size:
                iterator.append(batch)
                batch, b = [], 0

        return iterator

    @staticmethod
    def negtive_sampling(walks, node2id, sample_size=1e5):
        n2freq = {}
        for walk in walks:
            for node in walk:
                node = node2id[node]
                if node not in n2freq:
                    n2freq[node] = 1
                else:
                    n2freq[node] += 1

        freqs = np.array(list(n2freq.values())) ** 0.75
        freqs = freqs / sum(freqs)
        freqs = np.rint(freqs * sample_size).astype(int)

        sampling_pool = []
        for node, freq in enumerate(freqs):
            sampling_pool += [node] * freq

        return sampling_pool

    def train(self):
        for i in range(self.w):
            walks = self.random_walk.get_walks()
            sampling_pool = self.negtive_sampling(walks, self.g.node2id)

            for batch in self.get_iterator(walks, self.g.node2id, self.batch_size, self.k):
                u = [pair[0] for pair in batch]
                v = [pair[1] for pair in batch]
                neg_v = np.random.choice(sampling_pool, size=(len(v), self.neg_size))

                u = torch.LongTensor(u)
                v = torch.LongTensor(v)
                neg_v = torch.LongTensor(neg_v)

                loss = self.skip_gram.forward(u, v, neg_v)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_embeddings(self):
        torch.save(self.skip_gram.state_dict(), 'embeddings_metapath2vec.pt')

        return self.skip_gram.state_dict()['u_embedding.weight']
