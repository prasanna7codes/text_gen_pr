import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
try:
    model = load_model('text_gen_pr/next_word_lstm.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
    except Exception as e:
        st.error(f"Error predicting next word: {e}")
    return None

# Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("Please enter a valid sequence of words.")
    else:
        try:
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            st.write(f'Next word: {next_word}' if next_word else "Could not predict the next word.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
