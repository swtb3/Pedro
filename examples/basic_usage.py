import numpy as np
from pedro import Pedro
import ast
import pandas as pd

df = pd.read_csv("")

emb = df["embedding"].apply(ast.literal_eval).tolist()
emb = np.array(emb).astype("float32")
ped = Pedro(emb, k=100, method="pacmap", dr_kwargs={"n_neighbors": 30})

ped.plot()
