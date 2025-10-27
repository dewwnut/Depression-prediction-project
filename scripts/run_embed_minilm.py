import argparse
import logging
import sys 
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]  
)

logger = logging.getLogger("run_embed_minilm")

def safe_makdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    
def try_import_embedder():
    try:
        from src.embedder import MiniLMEmbedder
        logger.info("Imported MiniLMEmbedder from src.embedder.")
        return MiniLMEmbedder
    except Exception as e:
        logger.warning(f"Could not import src.embedder.MiniLMEmbedder: {e}. Falling back to internal embedder.")
        
        #minimal fallback embedder
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as se:
            logger.error("sentence-transformers is not installed. Please install it (pip install sentence-transformers)")
            raise
        
        class MiniLMEmbedderFallback:
            def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu', batch_size=128):
                self.model_name = model_name
                self.batch_size = batch_size
                self.device = device
                self.model = SentenceTransformer(model_name, device=device)
            
            def encode(self, texts, show_progress=True):
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    emb = self.model.encode(batch, show_progress_bar=show_progress, convert_to_numpy=True)
                    embeddings.append(emb)
                return np.vstack(embeddings)
            
        return MiniLMEmbedderFallback
    
def main(args):
    start_time = time.time()

    clean_csv = Path("C:/Users/Owner/depression-prediction/data/reddit_depression_small.csv")
    cache_dir = Path("C:/Users/Owner/depression-prediction/data/cache")
    cache_path = cache_dir / "embeddings_minilm_small.pkl"
    ids_path = cache_dir / "ids_minilm_small.pkl"

    if not clean_csv.exists():
        logger.error(f"Cleaned CSV file not found at: {clean_csv}")
        sys.exit(1)
        
    safe_makdir(cache_dir)
    
    df = pd.read_csv(clean_csv)
    if 'body' not in df.columns:
        logger.error("Input CSV must contain a 'body' column.")
        sys.exit(1)
        
    body = df['body'].astype(str).tolist()
    
    
    if cache_path.exists() and not args.force:
       logger.info(f"Embeddings cache already exists at {cache_path}. Use --force to recompute.")
       return

    EmbedderClass = try_import_embedder()
    embedder = EmbedderClass(model_name=args.model_name, device=args.device, batch_size=args.batch_size)

    try:
        logger.info(f"Encoding {len(body)} texts with model {args.model_name} (device={args.device}, batch_size={args.batch_size})")
        emb = embedder.encode(body, show_progress=True) if 'show_progress' in embedder.encode.__code__.co_varnames else embedder.encode(body)
    except TypeError:
        emb = embedder.encode(body)

    logger.info(f"Embeddings computed. shape={emb.shape}")

    joblib.dump(emb, cache_path)
    joblib.dump(df.index.to_list(), ids_path)

    elapsed = time.time() - start_time
    logger.info(f"Saved embeddings to {cache_path} and ids to {ids_path}. Time elapsed: {elapsed:.1f}s")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Compute & cache MiniLM embeddings")
    parser.add_argument('--clean_csv', type=str, default='data/cleaned/reddit_depression_cleaned.csv', help='Path to cleaned CSV with a `text` column')
    parser.add_argument('--cache_dir', type=str, default='data/cache', help='Directory to store cached embeddings')
    parser.add_argument('--emb_cache', type=str, default='embeddings_minilm.pkl', help='Embeddings cache filename')
    parser.add_argument('--ids_cache', type=str, default='ids_minilm.pkl', help='IDs cache filename')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Sentence-transformers model name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for encoding')
    parser.add_argument('--force', action='store_true', help='Force recompute embeddings even if cache exists')
    args = parser.parse_args()
    main(args)
