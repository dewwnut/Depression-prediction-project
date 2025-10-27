import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

# Example small LIWC-like dictionaries
SAD_WORDS = {"sad", "depressed", "hopeless", "cry", "lonely"}
ANX_WORDS = {"anxious", "nervous", "worry", "panic"}
SOCIAL_WORDS = {"friend", "family", "we", "us", "together"}
I_PRONOUNS = {"i", "me", "my", "mine"}

def tokenize(text):
    text = (text or "").lower()     # handle None/NaN
    tokens = re.findall(r"\b[a-z']+\b", text)
    return tokens

def liwc_like_features(text):
    tokens = tokenize(text)
    n_words = max(1, len(tokens))
    return {
        "pct_sad": sum(1 for t in tokens if t in SAD_WORDS) / n_words,
        "pct_anx": sum(1 for t in tokens if t in ANX_WORDS) / n_words,
        "pct_social": sum(1 for t in tokens if t in SOCIAL_WORDS) / n_words,
        "pct_I_pronouns": sum(1 for t in tokens if t in I_PRONOUNS) / n_words,
        "avg_word_len": sum(len(t) for t in tokens) / n_words,
    }

# --- Load data ---
df = pd.read_csv(r"C:/Users/Owner/depression-prediction/data/reddit_depression_small.csv")

# ensure columns exist and rename if needed
df = df.rename(columns={ "body": "text" })  

# drop rows with missing labels or coerce label
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# sample up to 1000 safely
n_samples = min(1000, len(df))
df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

# compute LIWC-style features
liwc_feats = df["text"].fillna("").apply(lambda t: pd.Series(liwc_like_features(t)))
df = pd.concat([df, liwc_feats], axis=1)

# TF-IDF (safe defaults)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["text"].fillna(""))

# scale LIWC features
liwc_cols = ["pct_sad","pct_anx","pct_social","pct_I_pronouns","avg_word_len"]
scaler = StandardScaler()
X_extra_scaled = scaler.fit_transform(df[liwc_cols].values)

# convert dense LIWC to sparse and concatenate
X_extra_sparse = csr_matrix(X_extra_scaled)
X = hstack([X_tfidf, X_extra_sparse], format="csr")

y = df["label"].values

# Train/test split (on sparse features)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))


background_sample = X_train[:200].toarray() if X_train.shape[0] > 200 else X_train.toarray()

masker = shap.maskers.Independent(background_sample)
explainer = shap.LinearExplainer(model, masker)

# compute shap values for a subset if X_test is large to save time/memory
X_test_subset = X_test[:200].toarray() if X_test.shape[0] > 200 else X_test.toarray()
shap_values = explainer.shap_values(X_test_subset)  # <- use the subset here

# feature names
tfidf_names = tfidf.get_feature_names_out()
feature_names = list(tfidf_names) + liwc_cols

# summary plot
shap.summary_plot(shap_values, X_test_subset, feature_names=feature_names, show=True)
