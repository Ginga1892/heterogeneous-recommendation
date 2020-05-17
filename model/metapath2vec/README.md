# metapth2vec

A metapath2vec (DeepWalk when homogeneous) implement based on PyTorch.

## Usage

```python
hen = HN().from_edge(edgelist)
network.get_vocab()

model = MetaPath2Vec(hen, metapath, walks_per_node, walk_len, window_size, emb_size, neg_size, batch_size, learning_rate)
model.train()

embeddings = model.get_embeddings()
```

## Follow Up

* API
* metapath2vec++