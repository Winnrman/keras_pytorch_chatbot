# Import necessary libraries
from model import create_model
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # Load the dataset
    df = pd.read_parquet('version1.parquet')

    # Preprocess the data
    prompts = df['prompt'].apply(preprocess_text).tolist()
    responses = df['response'].apply(preprocess_text).tolist()

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(prompts, responses, test_size=0.2, random_state=42)

    # Tokenize the data
    X_train, vocab_size, max_sequence_length, word_index = tokenize_data(X_train)
    X_valid, _, _, _ = tokenize_data(X_valid)

    # Prepare the data
    X_train_padded, y_train_padded = prepare_data(X_train, vocab_size, max_sequence_length)
    X_valid_padded, y_valid_padded = prepare_data(X_valid, vocab_size, max_sequence_length)

    # Create the model
    model = create_model(vocab_size, max_sequence_length)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train_padded, y_train_padded, epochs=100, validation_data=(X_valid_padded, y_valid_padded), batch_size=64)

    # Save the model
    model.save('chatbot_v1.keras')
