import os
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

DATA_DIR = Path(r"C:/Users/Owner/depression-prediction/data")
candidates = [
    DATA_DIR / "labels.npy",
    DATA_DIR / "labels.pkl",
    DATA_DIR / "labels.csv",
    DATA_DIR / "y.npy",
    DATA_DIR / "y.pkl",
    DATA_DIR / "y.csv",
]

found = False
for p in candidates:
    if p.exists():
        found = True
        print("Found labels file:", p)
        if p.suffix == ".npy":
            y = np.load(p)
        elif p.suffix == ".pkl":
            with open(p, "rb") as f:
                y = pickle.load(f)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            # try common column names
            for col in ("label","labels","y","target"):
                if col in df.columns:
                    y = df[col].values
                    break
            else:
                y = df.iloc[:,0].values
        print("Labels shape:", getattr(y, "shape", len(y)))
        print("First 20 labels:", list(y[:20]))
        try:
            print("Unique classes and counts:", dict(zip(*np.unique(y, return_counts=True))))
        except Exception as e:
            print("Could not compute uniques:", e)
if not found:
    print("No labels found in data/. Please create data/labels.npy (aligned with embeddings).")
