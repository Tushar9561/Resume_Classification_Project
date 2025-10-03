import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

# Load model and preprocessing
model = tf.keras.models.load_model("resume_classifier.h5")
tfidf_vector = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

st.title( Resume Classification App")
st.write("Upload or paste a resume and get its predicted category.")

# Input: Resume text
resume_input = st.text_area("Paste Resume Text Here:")

if st.button("Predict Category"):
    if resume_input.strip() != "":
        # Transform with TF-IDF
        resume_tfidf = tfidf_vector.transform([resume_input])
        resume_tfidf = resume_tfidf.toarray()  # Dense for NN

        # Predict
        prediction = model.predict(resume_tfidf)
        predicted_class = np.argmax(prediction, axis=1)
        category = le.inverse_transform(predicted_class)[0]

        st.success(f"Predicted Category: **{category}**")
    else:
        st.warning("Please enter some text before predicting.")

