from model import create_model
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

if __name__ == '__main__':

    if torch.backends.mps.is_available():        
        print("Metal is available. Neural Engine is being utilized.") #it is.
    else:
        print("Metal is not available. Neural Engine is not being utilized.")

    # Load the Parquet file into a DataFrame
    df = pd.read_parquet('dataset.parquet')

    # Preprocess the question column
    df['question_processed'] = df['question'].apply(preprocess_text)

    # Tokenize the data
    sequences, vocab_size, max_sequence_length, word_index = tokenize_data(df['question_processed'])

    # Split the data into training and validation sets for both X and y
    X_train, X_valid, y_train, y_valid = train_test_split(sequences, sequences, test_size=0.2, random_state=42)

    # Prepare the data
    X_train_padded, y_train_onehot = prepare_data(X_train, vocab_size, max_sequence_length)
    X_valid_padded, y_valid_onehot = prepare_data(X_valid, vocab_size, max_sequence_length)

    # Create the model
    model = create_model(vocab_size, max_sequence_length)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train_onehot, validation_data=(X_valid_padded, y_valid_onehot), epochs=5, batch_size=64)

    #view a summary of the model
    model.summary()

    # Save the model
    model.save('chatbot_model.keras')