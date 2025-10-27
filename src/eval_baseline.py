from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "data" / "reddit_depression_cleaned.csv"
tfidf_path = BASE_DIR.parent / "models" / "tfidf_vectorizer.joblib"
model_path = BASE_DIR.parent / "models" / "logistic_model.joblib"

# --------------------
# Load data
# --------------------
df = pd.read_csv(data_path)
df = df.dropna(subset=['label'])
df['body'] = df['body'].fillna('').astype(str)
df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)

X = df['body']
y = df['label']

# --------------------
# Train/test split
# --------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------
# Load saved model & vectorizer
# --------------------
vectorizer = joblib.load(tfidf_path)
model = joblib.load(model_path)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test_tfidf)

# --------------------
# Evaluate
# --------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
