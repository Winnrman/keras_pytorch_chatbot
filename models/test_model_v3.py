from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from config import max_sequence_length

# Load the Parquet file into a DataFrame
df = pd.read_parquet('daily_dialog.parquet')

# Preprocess the dialog, act, and emotion columns
dialogs = df['dialog'].apply(preprocess_text).tolist()
acts = df['act'].apply(preprocess_text).tolist()
emotions = df['emotion'].apply(preprocess_text).tolist()

# Combine acts and emotions into a single list
acts_emotions = [f"{act} {emotion}" for act, emotion in zip(acts, emotions)]

# Combine dialogues and act_emotion values into a single list
combined_data = [' '.join([str(item) for item in pair]) for pair in zip(dialogs, acts_emotions)]

# Tokenize the combined data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_data)
word_index = tokenizer.word_index

# Create reverse_word_index
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('chatbot_model.keras')

# Your testing logic here
# User input loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Tokenize the input
    tokenized_input = tokenizer.texts_to_sequences([user_input])[0]

    # Pad the tokenized input
    padded_input = pad_sequences([tokenized_input], maxlen=max_sequence_length, padding='pre')

    # Make prediction
    prediction = model.predict(padded_input)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Print the response
    print("AI:", reverse_word_index[predicted_index])
