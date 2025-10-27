#  Detecting Linguistic Markers of Depression Using NLP

A **Natural Language Processing (NLP)** project that predicts potential signs of **depression** from written text using **Machine Learning**.  
This model was trained on anonymized Reddit posts labeled for depressive and non-depressive content.

>  **Disclaimer:** This project is for **educational and research purposes only**. It is **not a diagnostic or clinical tool**.

---

##  Overview

The goal of this project is to explore how language patterns can reflect emotional states.  
By analyzing text data from Reddit, the model learns to detect linguistic cues often associated with depressive language.

### Objectives:
- Clean and preprocess text data
- Extract linguistic features using **TF-IDF**
- Train a **Logistic Regression** model for binary classification
- Compare with **MiniLM sentence embeddings**
- Interpret model predictions using **SHAP**

---

##  Project Structure
```
depression-prediction/
│
├── data/
│ ├── reddit_depression_dataset.csv
│ ├── reddit_depression_cleaned.csv
│ └── reddit_depression_small.csv
│
├── models/
│ ├── baseline_pipeline.joblib
│ ├── logistic_model.joblib
│ ├── rf_minilm.pkl
│ └── tfidf_vectorizer.joblib
│
├── notebooks/
│ └── EDA.ipynb
│
├── scripts/
│ ├── create_small_dataset.py
│ ├── run_embed_minilm.py
│ └── run_train_minilm.py
│
├── src/
│ ├── clean_data.py
│ ├── create_labels_from_csv.py
│ ├── embedder.py
│ ├── train_baseline.py
│ ├── train_minilm.py
│ ├── eval_baseline.py
│ ├── eval_minilm.py
│ ├── eval_threshold.py
│ ├── liwc_like_features.py
│ ├── inspect_embeddings_joblib.py
│ └── inspect_labels.py
│
├── README.md
└── requirements.txt

```

##  Methodology

### 1. **Data Preprocessing**
- Removed URLs, emojis, special characters, and extra spaces  
- Lowercased all text and removed stopwords  
- Balanced dataset using `class_weight='balanced'`  

### 2. **Feature Extraction**
- **TF-IDF Vectorization:** Captures term frequency importance  
- **MiniLM Embeddings:** Captures deeper semantic meaning  

### 3. **Model Training**

- **Baseline:** `LogisticRegression(max_iter=1000, class_weight='balanced')`  
  - Trained on **TF-IDF features**

- **Advanced:** `RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced')`  
  - Trained on **MiniLM embeddings**



### 4. **Evaluation**

Model performance was evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC AUC**.  
Two models were trained and compared to measure the impact of different text representations.

| Model | Text Representation | Accuracy | Precision | Recall | F1-score | ROC AUC |
|:------|:------------------|:--------:|:---------:|:------:|:---------:|:-------:|
| **Baseline:** Logistic Regression | TF-IDF | **0.91** | 0.92 | 0.91 | 0.92 | – |
| **Advanced:** Random Forest | MiniLM Embeddings | **0.91** | 0.91 | 0.91 | 0.91 | 0.94 |

> The **Random Forest model** trained on **MiniLM embeddings** slightly outperforms the baseline in terms of class separation (ROC AUC = 0.94), indicating that semantic embeddings capture deeper linguistic and contextual cues associated with depressive language, even though weighted accuracy and F1-score are similar to the baseline Logistic Regression model.



---

##  Explainability with SHAP

To understand *why* the model predicts “depressive” or “non-depressive,”  
SHAP (SHapley Additive exPlanations) was used for interpretability.

- Visualized most influential words  
- Verified that features like **“worthless,” “tired,” “lonely,”** increased depression likelihood  
- Ensured transparency and ethical interpretability

---

##  Technologies Used

| Category | Libraries |
|-----------|------------|
| Language | Python 3.11 |
| NLP | NLTK, spaCy, scikit-learn |
| Machine Learning | scikit-learn |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Utilities | Pandas, NumPy |

---

##  Future Improvements

- Fine-tune transformer models like **RoBERTa-base** or **DistilBERT**
- Perform **hyperparameter tuning** for improved accuracy
- Expand to multi-class classification (e.g., emotional states)

---

##  Ethical Statement

This project explores the intersection of language and mental health through **ethical AI**.  
All data was publicly available and anonymized.  
It must **never** replace professional assessment or therapy.  



##  Example Use

```python
from joblib import load

# Load model and vectorizer
model = load("models/baseline_model.pkl")
vectorizer = load("models/vectorizer.pkl")

# Predict from new text
text = ["I feel so hopeless lately, nothing excites me anymore."]
features = vectorizer.transform(text)
prediction = model.predict(features)

print("Prediction:", "Depressive" if prediction[0] == 1 else "Non-depressive")
```



##  Setup & Run Instructions

Follow these steps to set up the project on your local machine.

### 1 Clone the Repository

```bash
git clone https://github.com/dewwnut/depression-prediction.git
cd depression-prediction
```

### 2 Create and Activate a Virtual Environment

Using conda:
```bash
conda create -n depression-prediction python=3.11 -y
conda activate depression-prediction
```
Or using venu:
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On macOS/Linux
```

### 3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4 Run the Baseline Model (TF-IDF + Logistic Regression)
The baseline pipeline trains and evaluates a Logistic Regression model using TF-IDF features.

```bash
python src/train_baseline.py
```
To evaluate its performance:

```bash
python src/eval_baseline.py
```
### 5 Run the MiniLM Model (Semantic Embeddings + Random Forest)

Step 1 — Create a smaller dataset (optional for faster testing)
```bash
python scripts/create_small_dataset.py
```

Step 2 — Generate MiniLM Embeddings
```bash
python scripts/run_embed_minilm.py
```

Step 3 — Train the Model
```bash
python scripts/run_train_minilm.py
```

Step 4 — Evaluate the MiniLM Model
```bash
python src/eval_minilm.py
```
This will train a Random Forest classifier using MiniLM sentence embeddings and save results under models/.

### 6  (Optional) Explore EDA Notebook
If you want to view data exploration and visualizations:
```bash
jupyter notebook notebooks/EDA.ipynb
```



## Model Artifacts

After running the scripts, you’ll find:

-Baseline Model: models/logistic_model.joblib

-MiniLM Model: models/rf_minilm.pkl

-TF-IDF Vectorizer: models/tfidf_vectorizer.joblib



## Troubleshooting

If you face memory or speed issues while generating embeddings, you can:

Use a smaller subset of the data via create_small_dataset.py

Run on CPU only (default; no GPU required)

Close other applications to free memory during embedding generation


##  Acknowledgments

- **Dataset:** [Reddit Depression Dataset (Kaggle)](https://www.kaggle.com/datasets)  
- **Inspired by:** Academic research on NLP and mental health  
- **Author:** [Nada Badran](https://github.com/dewwnut)

---