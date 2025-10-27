import pandas as pd
import numpy as np
from pathlib import Path


CSV_PATH = Path("C:/Users/Owner/depression-prediction/data/reddit_depression_small.csv")   
LABEL_COL = "label"                         
OUT = Path("C:/Users/Owner/depression-prediction/data/labels.npy")

if not CSV_PATH.exists():
    raise SystemExit(f"CSV not found: {CSV_PATH}. Put your CSV here or change CSV_PATH.")

df = pd.read_csv(CSV_PATH)
if LABEL_COL not in df.columns:
    print("Available columns:", df.columns.tolist())
    raise SystemExit(f"Column '{LABEL_COL}' not found. Update LABEL_COL in the script.")

labels = df[LABEL_COL].values
print("Loaded labels from CSV. Shape:", labels.shape)
print("Unique classes and counts:", dict(zip(*np.unique(labels, return_counts=True))))


emb_path = Path(r"C:/Users/Owner/depression-prediction/data/cache/embeddings_minilm_small.pkl")
import joblib, numpy as np
obj = joblib.load(emb_path)
X = np.asarray(getattr(obj, "data", obj)) if hasattr(obj, "data") else np.asarray(obj)
print("Embeddings shape (detected):", X.shape)

if len(labels) != X.shape[0]:
    raise SystemExit(f"Length mismatch! labels: {len(labels)}, embeddings: {X.shape[0]}. They must match and be in the same order.")

np.save(OUT, labels)
print("Saved labels to", OUT)
