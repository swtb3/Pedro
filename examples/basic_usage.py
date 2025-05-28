import numpy as np
from pedro import Pedro
import ast
import pandas as pd
# synthetic demo data
# emb = np.random.rand(200, 1024).astype("float32")
df = pd.read_csv("C:\\Users\\sbay\\Documents\\workspace\\coding\\proposals_testing\\lightweight_proposals_scripts\\data\\embedding_batches\\embeddings_batch_000.csv")


# Parse the embeddings column (assumes it's stringified lists)
emb = df["embedding"].apply(ast.literal_eval).tolist()
emb = np.array(emb).astype("float32")
ped = Pedro(emb, k=100, method="pacmap", dr_kwargs={"n_neighbors": 30})

ped.plot()
