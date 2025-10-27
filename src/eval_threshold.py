import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Paths
EMB_PATH = Path(r"C:/Users/Owner/depression-prediction/data/cache/embeddings_minilm_small.pkl")
LABELS_PATH = Path("data/labels.npy")
MODEL_PATH = Path("models/rf_minilm.pkl")

# Load embeddings
obj = joblib.load(EMB_PATH)
X = getattr(obj, "data", obj) if hasattr(obj, "data") else obj
X = np.asarray(X)

# Load labels
y = np.load(LABELS_PATH)

# Load model
model = joblib.load(MODEL_PATH)

# Split test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]  
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    print(f"\n--- Threshold = {t} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
