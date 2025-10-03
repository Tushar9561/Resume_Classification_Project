{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2851f18",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\keras\\src\\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-10-03 16:04:20.728 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.896 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.905 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.906 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Session state does not function when running a script without `streamlit run`\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.915 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.916 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-03 16:04:20.917 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "# Load model and preprocessing\n",
    "model = tf.keras.models.load_model(\"resume_classifier.h5\")\n",
    "tfidf_vector = joblib.load(\"tfidf_vectorizer.pkl\")\n",
    "le = joblib.load(\"label_encoder.pkl\")\n",
    "\n",
    "st.title(\" Resume Classification App\")\n",
    "st.write(\"Upload or paste a resume and get its predicted category.\")\n",
    "\n",
    "# Input: Resume text\n",
    "resume_input = st.text_area(\"Paste Resume Text Here:\")\n",
    "\n",
    "if st.button(\"Predict Category\"):\n",
    "    if resume_input.strip() != \"\":\n",
    "        # Transform with TF-IDF\n",
    "        resume_tfidf = tfidf_vector.transform([resume_input])\n",
    "        resume_tfidf = resume_tfidf.toarray()  # Dense for NN\n",
    "\n",
    "        # Predict\n",
    "        prediction = model.predict(resume_tfidf)\n",
    "        predicted_class = np.argmax(prediction, axis=1)\n",
    "        category = le.inverse_transform(predicted_class)[0]\n",
    "\n",
    "        st.success(f\" Predicted Category: **{category}**\")\n",
    "    else:\n",
    "        st.warning(\" Please enter some text before predicting.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23a9375e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
