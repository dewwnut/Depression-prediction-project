import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


logger = logging.getLogger("run_train_minilm")


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_embeddings(emb_path: Path):
    if not emb_path.exists():
        logger.error(f"Embeddings not found at {emb_path}")
        return None
    return joblib.load(emb_path)


def main(args):
    start_time = time.time()

    clean_csv = Path(args.clean_csv)
    emb_cache = Path(args.emb_cache)
    ids_cache = Path(args.ids_cache)
    models_dir = Path(args.models_dir)
    safe_mkdir(models_dir)

    if not clean_csv.exists():
        logger.error(f"Clean CSV not found: {clean_csv}")
        sys.exit(1)

    df = pd.read_csv(clean_csv)
    if 'label' not in df.columns:
        logger.error("Expected column 'label' in cleaned CSV")
        sys.exit(1)

    labels = df['label'].values

    emb = load_embeddings(emb_cache)
    if emb is None or args.recompute:
        
        logger.info("Embeddings missing or recompute requested — invoking run_embed_minilm script")
        import subprocess
        import sys as _sys
        cmd = [_sys.executable, 'scripts/run_embed_minilm.py', '--clean_csv', str(clean_csv), '--cache_dir', str(args.cache_dir), '--emb_cache', emb_cache.name]
        if args.force_embed:
            cmd.append('--force')
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            logger.error("Failed to compute embeddings via scripts/run_embed_minilm.py")
            sys.exit(1)
        emb = load_embeddings(emb_cache)
        if emb is None:
            logger.error("Embeddings still not available after running embed script")
            sys.exit(1)

    logger.info(f"Loaded embeddings shape: {emb.shape}")

    # Optional dimensionality reduction
    svd = None
    X = emb
    if args.use_svd:
        logger.info(f"Applying TruncatedSVD (n_components={args.svd_components})")
        svd = TruncatedSVD(n_components=args.svd_components, random_state=args.random_state)
        X = svd.fit_transform(emb)
        joblib.dump(svd, models_dir / args.svd_out)
        logger.info(f"Saved SVD to {models_dir / args.svd_out}")

    # Align X and labels length
    if X.shape[0] != labels.shape[0]:
        logger.warning(f"Embeddings rows ({X.shape[0]}) and labels ({labels.shape[0]}) mismatch. Attempting to align using ids cache if present.")
        if ids_cache.exists():
            ids = joblib.load(ids_cache)
            # ids are original dataframe indices; create mask
            idx_map = {i: pos for pos, i in enumerate(ids)}
            mask = [i in idx_map for i in range(len(df))]
            X_aligned = []
            y_aligned = []
            for i, orig_idx in enumerate(ids):
                if orig_idx < len(df):
                    X_aligned.append(X[i])
                    y_aligned.append(labels[orig_idx])
            X = np.vstack(X_aligned)
            labels = np.array(y_aligned)
            logger.info(f"Aligned shapes after using ids: X={X.shape}, labels={labels.shape}")
        else:
            logger.error("Cannot align embeddings and labels; please check your cache and cleaned CSV")
            sys.exit(1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=args.test_size, stratify=labels, random_state=args.random_state)

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train classifier
    clf = LogisticRegression(max_iter=args.max_iter, class_weight='balanced', n_jobs=args.n_jobs, random_state=args.random_state)
    logger.info("Fitting LogisticRegression")
    clf.fit(X_train, y_train)

    # Save model
    model_path = models_dir / args.model_out
    joblib.dump(clf, model_path)
    logger.info(f"Saved classifier to {model_path}")

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'classification_report': report,
        'confusion_matrix': cm,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'model_path': str(model_path),
        'svd_path': str(models_dir / args.svd_out) if svd is not None else None,
    }

    metrics_path = models_dir / args.metrics_out
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"Training complete. Metrics saved to {metrics_path}. Time elapsed: {elapsed:.1f}s")
    logger.info("Classification report (text):")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Train classifier on MiniLM embeddings")
    parser.add_argument('--clean_csv', type=str, default='data/cleaned/reddit_depression_cleaned.csv')
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--emb_cache', type=str, default='data/cache/embeddings_minilm.pkl')
    parser.add_argument('--ids_cache', type=str, default='data/cache/ids_minilm.pkl')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--recompute', action='store_true', help='If embeddings are missing, recompute by calling the embed script')
    parser.add_argument('--force_embed', action='store_true', help='When recomputing embeddings, force overwrite existing cache')
    parser.add_argument('--use_svd', action='store_true', help='Apply TruncatedSVD before training')
    parser.add_argument('--svd_components', type=int, default=128)
    parser.add_argument('--svd_out', type=str, default='svd_minilm.pkl')
    parser.add_argument('--model_out', type=str, default='minilm_logreg.pkl')
    parser.add_argument('--metrics_out', type=str, default='metrics_minilm.json')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Normalize some paths
    if isinstance(args.emb_cache, str) and not args.emb_cache.startswith('data/'):
        # allow passing just filename
        args.emb_cache = args.emb_cache

    try:
        main(args)
    except Exception as e:
        logger.exception(f"Unhandled exception during training: {e}")
        sys.exit(1)
