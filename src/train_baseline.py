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
models_dir.mkdir(parents=True, exist_ok=True)

# --------------------
# Load + clean data
# --------------------
df = pd.read_csv(data_path)
print("Loaded dataset:", df.shape)

# Drop rows without labels and make sure body is str and label is numeric
df = df.dropna(subset=['label'])
df['body'] = df['body'].fillna('').astype(str)
df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)

print("Missing labels after drop:", df['label'].isna().sum())
print(df['label'].value_counts(dropna=False))

# OPTIONAL: sample for faster dev 
#df = df.sample(n=200000, random_state=42).reset_index(drop=True)
#print("Sampled dataset:", df.shape)

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
    solver='saga',            
    penalty='l2',
    class_weight='balanced',  
    random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=2000, solver='saga', penalty='l2', class_weight='balanced', random_state=42))
])

# --------------------
# Train
# --------------------
print("Fitting pipeline (this may take a while)...")
pipeline.fit(X_train, y_train)

# --------------------
# Save pipeline 
# --------------------
pipeline_path = models_dir / "baseline_pipeline.joblib"
joblib.dump(pipeline, pipeline_path)
print(f"Saved pipeline to: {pipeline_path}")


tfidf_path = models_dir / "tfidf_vectorizer.joblib"
model_path = models_dir / "logistic_model.joblib"
joblib.dump(pipeline.named_steps['tfidf'], tfidf_path)
joblib.dump(pipeline.named_steps['clf'], model_path)

print(f"Training complete. Saved pipeline, tfidf, and model to {models_dir}")
