import argparse
from pathlib import Path
import numpy as np
import joblib
import pickle
import os
from collections import Counter
import json
import pandas as pd
import smote

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score


try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# ---------- Config / paths ----------
EMB_PATH = Path(r"C:/Users/Owner/depression-prediction/data/cache/embeddings_minilm_small.pkl")
DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "labels.npy"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helper functions ----------
def _to_numpy(obj):
    """Convert many common container types to numpy array."""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj
    
    if hasattr(obj, "data"):
        try:
            return _np.asarray(obj.data)
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        return _np.asarray(obj)
    if isinstance(obj, dict):
        for key in ("embeddings", "X", "vectors", "arr"):
            if key in obj:
                return _np.asarray(obj[key])
        # fallback: take first array-like with ndim >= 2
        for v in obj.values():
            try:
                a = _np.asarray(v)
                if getattr(a, "ndim", 1) >= 2:
                    return a
            except Exception:
                pass
    # final fallback
    return _np.asarray(obj)

def load_embeddings(path: Path):
    """Try joblib, pickle, np.load in that order and return numpy.ndarray."""
    path = Path(path)
    # try joblib
    try:
        obj = joblib.load(path)
        arr = _to_numpy(obj)
        return arr
    except Exception:
        pass
    # try pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return _to_numpy(obj)
    except Exception:
        pass
    # try numpy
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        pass
    raise ValueError(f"Could not load embeddings from {path} — unknown format.")

def find_labels(data_dir: Path):
    candidates = [
        data_dir / "labels.npy",
        data_dir / "labels.pkl",
        data_dir / "labels.csv",
        data_dir / "y.npy",
        data_dir / "y.pkl",
        data_dir / "y.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".npy":
                return np.load(p)
            if p.suffix == ".pkl":
                with open(p, "rb") as f:
                    return pickle.load(f)
            if p.suffix == ".csv":
                df = pd.read_csv(p)
                for col in ("label","labels","y","target"):
                    if col in df.columns:
                        return df[col].values
                return df.iloc[:,0].values
    return None

def save_splits(X_train, X_test, y_train, y_test):
    np.save(SPLITS_DIR / "X_train.npy", X_train)
    np.save(SPLITS_DIR / "X_test.npy", X_test)
    np.save(SPLITS_DIR / "y_train.npy", y_train)
    np.save(SPLITS_DIR / "y_test.npy", y_test)

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = None
    return metrics

# ---------- main ----------
def main(
    test_size=0.2,
    random_state=42,
    use_smote=False,
    out_model="models/rf_minilm.pkl",
    n_iter=6,
    cv=3,
):
    print("Loading embeddings from:", EMB_PATH)
    X = load_embeddings(EMB_PATH)
    X = np.asarray(X)
    print("Embeddings shape:", X.shape, "dtype:", X.dtype)

    print("Looking for labels in data/ ...")
    y = find_labels(DATA_DIR)
    if y is None:
        raise FileNotFoundError(f"No labels found in {DATA_DIR}. Please create data/labels.npy aligned to embeddings.")
    y = np.asarray(y)
    print("Labels shape:", y.shape)
    if len(y) != X.shape[0]:
        raise ValueError(f"Label count ({len(y)}) != embedding count ({X.shape[0]}). They must align in order and length.")

    print("Class distribution:", Counter(y))

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    save_splits(X_train, X_test, y_train, y_test)
    print("Saved splits to", SPLITS_DIR)

    # SMOTE option
    if use_smote:
        if not IMBLEARN_AVAILABLE:
            print("WARNING: imbalanced-learn not available. Install it to use SMOTE.")
        else:
            print("Applying SMOTE to training set...")
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print("After SMOTE:", Counter(y_train))

    # Build pipeline: scaler + classifier
    # RandomForest generally doesn't need scaling, but we include scaler so pipeline can be swapped easily.
    rf = RandomForestClassifier(n_jobs=-1, random_state=random_state, class_weight="balanced")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])

    # Hyperparameter search
    param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 20, 50],
        "clf__min_samples_leaf": [1, 2, 5],
    }

    print("Starting RandomizedSearchCV (this may take some time)...")
    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        verbose=2,
        n_jobs=-1,
        random_state=random_state,
    )

    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    print("Best params:", rs.best_params_)

    # Evaluate on test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    # Save model
    out_model = Path(out_model)
    joblib.dump(best, out_model)
    print("Saved model to", out_model)

    # Save results (metrics + params)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    results = {
        "model_path": str(out_model),
        "embeddings_path": str(EMB_PATH),
        "labels_path": str(LABELS_PATH),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_distribution_train": dict(zip(*np.unique(y_train, return_counts=True))),
        "class_distribution_test": dict(zip(*np.unique(y_test, return_counts=True))),
        "best_params": rs.best_params_,
        "metrics": metrics,
    }
    
    def convert_numpy(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    
    
    results_path = RESULTS_DIR / "rf_minilm_results.json"
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, default=convert_numpy)

    
    flat = {
        "model": str(out_model.name),
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "class_train_0": results["class_distribution_train"].get(0, 0),
        "class_train_1": results["class_distribution_train"].get(1, 0),
        "class_test_0": results["class_distribution_test"].get(0, 0),
        "class_test_1": results["class_distribution_test"].get(1, 0),
        **{f"param__{k}": v for k, v in rs.best_params_.items()},
        **{f"metric__{k}": v for k, v in metrics.items()},
    }
    csv_path = RESULTS_DIR / "rf_minilm_results.csv"
    # Append or create
    df_row = pd.DataFrame([flat])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)
    print("Saved results to", results_path, "and", csv_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE to training set (requires imbalanced-learn).")
    parser.add_argument("--model-out", default=str(MODELS_DIR / "rf_minilm.pkl"), help="Output path for trained model")
    parser.add_argument("--n-iter", type=int, default=6, help="n_iter for RandomizedSearchCV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(
        test_size=args.test_size,
        random_state=args.random_state,
        use_smote=args.smote,
        out_model=args.model_out,
        n_iter=args.n_iter,
    )
