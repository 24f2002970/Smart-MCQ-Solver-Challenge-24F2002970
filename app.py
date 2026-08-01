import streamlit as st 
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Configuration
# -------------------------------
MAX_LENGTH = 300

st.set_page_config(
    page_title="Smart MCQ Solver",
    layout="centered"
)


# -------------------------------
# Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("./model/model4_0_758.keras")
    tokenizer = joblib.load("./model/tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()


# -------------------------------
# UI
# -------------------------------

st.title("Smart MCQ Solver")
st.write("Enter the question and its five options.")

prompt= st.text_area("Question")

A = st.text_input("Option A")
B = st.text_input("Option B")
C = st.text_input("Option C")
D = st.text_input("Option D")
E = st.text_input("Option E")


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Answer"):

    if prompt.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    combined = (
        f"{prompt} "
        f"[SEP] {A} "
        f"[SEP] {B} "
        f"[SEP] {C} "
        f"[SEP] {D} "
        f"[SEP] {E}"
    )


    sequence = tokenizer.texts_to_sequences([combined])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    prediction = model.predict(padded, verbose=0)[0]

    labels = ["A", "B", "C", "D", "E"]

    top3_idx = np.argsort(prediction)[::-1][:3]

    st.success("Prediction Complete")

    st.subheader("Top 3 Predictions")

    for rank, idx in enumerate(top3_idx, start=1):
        st.write(
            f"**{rank}. Option {labels[idx]}** "
            f"({prediction[idx]*100:.2f}%)"
        )

    st.divider()

    st.subheader("Confidence for all options")

    for label, score in zip(labels, prediction):
        st.progress(float(score))
        st.write(f"Option {label}: {score*100:.2f}%")