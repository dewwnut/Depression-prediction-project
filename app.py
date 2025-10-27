import streamlit as st
from joblib import load

# --- Page Configuration ---
st.set_page_config(page_title="Depression Prediction App", page_icon="🧠", layout="centered")

# --- App Title & Description ---
st.title("🧠 Depression Prediction from Text")
st.write(
    "This web app uses a machine learning model trained on Reddit posts to predict "
    "whether a piece of writing might contain linguistic patterns associated with depression."
)

# --- Disclaimer ---
st.markdown("""
> ⚠️ **Disclaimer:**  
> This tool is for **educational and research purposes only**.  
> It is **not a diagnostic or mental health tool**, and its predictions should **not be used for medical or psychological evaluation**.  
> If you or someone you know is struggling, please seek help from a qualified professional or local mental health resource.
""")

st.divider()

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_artifacts():
    model = load("models/logistic_model.joblib")
    vectorizer = load("models/tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- User Input ---
st.subheader("💬 Analyze Your Text")
user_input = st.text_area("Enter your text below:", height=150)

# --- Predict Button ---
if st.button("🔍 Analyze"):
    if user_input.strip():
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]
        label = "Non-Depressive" if prediction == 0 else "Possibly Depressive"

        st.markdown(f"### Result: {label}")
    else:
        st.warning("Please enter some text before clicking Analyze.")

# --- Footer ---
st.markdown("---")
st.caption("Created by Nada Badran | Depression Prediction Project (NLP, Machine Learning)")
