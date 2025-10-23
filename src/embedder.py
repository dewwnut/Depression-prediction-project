from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from pathlib import Path

class MiniLMEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu', batch_size=128):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts, cache_path=None):
        if cache_path and Path(cache_path).exists():
            return joblib.load(cache_path)
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        if cache_path:
            joblib.dump(embeddings, cache_path)
        return embeddings
