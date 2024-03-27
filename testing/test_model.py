from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from config import max_sequence_length

# Load the Parquet file into a DataFrame
df = pd.read_parquet('daily_dialog.parquet')

# Preprocess the question column
df['question_processed'] = df['question'].apply(preprocess_text)

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['question_processed'])
word_index = tokenizer.word_index

# Create reverse_word_index
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('chatbot_model.keras')

# Your testing logic here
# For example, you can use the code from the previous message to test the model interactively

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
