from model import create_model
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from keras.utils import to_categorical

if __name__ == '__main__':
    # Load the dataset
    # Load the dataset
    df = pd.read_parquet('daily_dialog.parquet')

    # Select a subset of 200 rows for faster training
    df_subset = df.head(50)

    # Preprocess the dialog and act columns
    dialogs = df_subset['dialog'].apply(preprocess_text).tolist()
    acts = df_subset['act'].apply(preprocess_text).tolist()
    emotions = df_subset['emotion'].apply(preprocess_text).tolist()

    # Combine acts and emotions into a single list
    acts_emotions = [f"{act} {emotion}" for act, emotion in zip(acts, emotions)]

    # Combine dialogues and act_emotion values into a single list
    combined_data = [' '.join([str(item) for item in pair]) for pair in zip(dialogs, acts_emotions)]

    # Tokenize the combined data
    sequences, vocab_size, max_sequence_length, word_index = tokenize_data(combined_data)

    # Convert act/emotion labels to one-hot encoded vectors using the new word_index
    acts_emotions_onehot = []
    for act_emotion in acts_emotions:
        act_emotion_tokens = act_emotion.split()
        act_emotion_indices = [word_index[word] for word in act_emotion_tokens if word in word_index]
        act_emotion_onehot = to_categorical(act_emotion_indices, num_classes=vocab_size)
        acts_emotions_onehot.append(act_emotion_onehot)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(sequences, acts_emotions_onehot, test_size=0.2, random_state=42)

    # Prepare the data
    X_train_padded, y_train_onehot = prepare_data(X_train, vocab_size, max_sequence_length)
    X_valid_padded, y_valid_onehot = prepare_data(X_valid, vocab_size, max_sequence_length)

    # Create the model
    model = create_model(vocab_size, max_sequence_length)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_padded, np.array(y_train_onehot), validation_data=(X_valid_padded, np.array(y_valid_onehot)), epochs=4, batch_size=32)

    # Save the model
    model.save('chatbot_model.keras')

    #evaluate the model
    model.evaluate(X_valid_padded, np.array(y_valid_onehot)) #returns loss and accuracy

    #display a plot of the training history
    import matplotlib.pyplot as plt

    history = history.history
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
