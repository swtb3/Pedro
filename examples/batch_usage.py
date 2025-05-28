import numpy as np
import pandas as pd
import ast
from pathlib import Path

from pedro import Pedro

#For when you have batches of output embedding files

csv_folder = Path("")


csv_files = list(csv_folder.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_folder}")

df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df)} rows from {len(csv_files)} files.")

emb = df["embedding"].apply(ast.literal_eval).tolist()
emb = np.array(emb).astype("float32")

ped = Pedro(emb, k=250, method="pacmap", dr_kwargs={"n_neighbors": 30})
ped.plot()
