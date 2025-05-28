

def pacmap_reduce(embeddings, **kwargs):
    import pacmap
    reducer = pacmap.PaCMAP(**kwargs)
    return reducer.fit_transform(embeddings)

def umap_reduce(embeddings, **kw):
    import umap
    return umap.UMAP(**kw).fit_transform(embeddings)


def tsne_reduce(embeddings, **kw):
    from sklearn.manifold import TSNE
    return TSNE(**kw).fit_transform(embeddings)
