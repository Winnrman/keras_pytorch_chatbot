from model import create_model
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':

    # Load the Parquet file into a DataFrame
    df = pd.read_parquet('counseling.parquet')

    # Preprocess the question and answer columns
    df['question_processed'] = df['questionTitle'].apply(preprocess_text)
    df['answer_processed'] = df['answer'].apply(preprocess_text)

    # Tokenize the questions and answers
    question_sequences, question_vocab_size, question_max_sequence_length, question_word_index = tokenize_data(df['question_processed'])
    answer_sequences, answer_vocab_size, answer_max_sequence_length, answer_word_index = tokenize_data(df['answer_processed'])

    # Split the data into training and validation sets for questions and answers
    X_train_question, X_valid_question, y_train_answer, y_valid_answer = train_test_split(
        question_sequences, answer_sequences, test_size=0.2, random_state=42
    )

    # Prepare the data
    X_train_question_padded, y_train_answer_onehot = prepare_data(X_train_question, question_vocab_size, question_max_sequence_length)
    X_valid_question_padded, y_valid_answer_onehot = prepare_data(X_valid_question, question_vocab_size, question_max_sequence_length)

    # Create the model
    model = create_model(question_vocab_size, question_max_sequence_length)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_question_padded, y_train_answer_onehot, validation_data=(X_valid_question_padded, y_valid_answer_onehot), epochs=21, batch_size=64)

    # Save the model
    model.save('chatbot_model_simple_dataset.keras')
