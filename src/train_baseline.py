from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "data" / "reddit_depression_cleaned.csv"
models_dir = BASE_DIR.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)  # create models/ if it doesn't exist

if not data_path.exists():
    raise FileNotFoundError(f"File not found at: {data_path}")

# --------------------
# Load + basic cleaning
# --------------------
df = pd.read_csv(data_path)
print("Loaded dataset:", df.shape)

# Drop rows without labels and make sure body is str and label is numeric
df = df.dropna(subset=['label'])
df['body'] = df['body'].fillna('').astype(str)
df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)

print("Missing labels after drop:", df['label'].isna().sum())
print(df['label'].value_counts(dropna=False))

# OPTIONAL: sample for faster dev (uncomment during experimentation)
# df = df.sample(n=200000, random_state=42).reset_index(drop=True)
# print("Sampled dataset:", df.shape)

X = df['body']
y = df['label']

# --------------------
# Train/test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train/test sizes:", X_train.shape, X_test.shape)

# --------------------
# Pipeline: tfidf -> logistic
# --------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

clf = LogisticRegression(
    max_iter=2000,
    solver='saga',            # works well with sparse input
    penalty='l2',
    class_weight='balanced',  # optional: helps with class imbalance
    random_state=42
)

pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", clf)
])

# --------------------
# Train
# --------------------
print("Fitting pipeline (this may take a while)...")
pipeline.fit(X_train, y_train)

# --------------------
# Predict & evaluate
# --------------------
y_pred = pipeline.predict(X_test)   # pass raw text — pipeline handles transform

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --------------------
# Inspect top features (optional)
# --------------------
try:
    vect = pipeline.named_steps['tfidf']
    clf_fitted = pipeline.named_steps['clf']
    feature_names = np.array(vect.get_feature_names_out())
    coefs = clf_fitted.coef_[0]
    top_n = 20
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    print("\nTop features associated with class=1 (depressed):")
    for i in top_pos_idx:
        print(f"{feature_names[i]} ({coefs[i]:.4f})")

    print("\nTop features associated with class=0 (not depressed):")
    for i in top_neg_idx:
        print(f"{feature_names[i]} ({coefs[i]:.4f})")
except Exception as e:
    print("Could not extract feature importances:", e)

# --------------------
# Save pipeline (recommended)
# --------------------
pipeline_path = models_dir / "baseline_pipeline.joblib"
joblib.dump(pipeline, pipeline_path)
print(f"Saved pipeline to: {pipeline_path}")

# If you still want separate objects:
tfidf_path = models_dir / "tfidf_vectorizer.joblib"
model_path = models_dir / "logistic_model.joblib"
joblib.dump(pipeline.named_steps['tfidf'], tfidf_path)
joblib.dump(pipeline.named_steps['clf'], model_path)
print(f"Also saved tfidf to: {tfidf_path} and model to: {model_path}")
