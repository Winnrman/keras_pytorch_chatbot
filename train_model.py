# Import necessary libraries
from model import create_model
from preprocess_data import preprocess_text, tokenize_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':

    epochs = 50

    # Load the dataset
    df = pd.read_parquet('version1.parquet')

    # Preprocess the data
    prompts = df['prompt'].apply(preprocess_text).tolist() #turn the prompts into a list after processing them
    responses = df['response'].apply(preprocess_text).tolist() #turn the prompts into a list after processing them

    # Split the data into training and validation sets (80% training, 20% validation). 42 random_state means the split will be the same with different tests.
    X_train, X_valid, y_train, y_valid = train_test_split(prompts, responses, test_size=0.2, random_state=42)

    # Tokenize the data
    X_train, vocab_size, max_sequence_length, word_index = tokenize_data(X_train)
    X_valid, _, _, _ = tokenize_data(X_valid) #the 3 values which are not X_valid are given by the tokenize_data function. 

    # Prepare the data
    X_train_padded, y_train_padded = prepare_data(X_train, vocab_size, max_sequence_length)
    X_valid_padded, y_valid_padded = prepare_data(X_valid, vocab_size, max_sequence_length)

    model = create_model(vocab_size, max_sequence_length)
    optimizer = Adam(learning_rate=0.01)  # Set initial learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define a learning rate schedule
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr  # Keep initial learning rate for first 10 epochs
        else:
            return lr * 0.1  # Decrease learning rate by a factor of 10 after 10 epochs

# Create a learning rate scheduler callback
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Train the model with the learning rate scheduler callback
    # Train the model with the learning rate scheduler callback
    history = model.fit(X_train_padded, y_train_padded, epochs=epochs, 
                        validation_data=(X_valid_padded, y_valid_padded), 
                        callbacks=[lr_schedule])


    # Create the model
    # model = create_model(vocab_size, max_sequence_length)

    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # Fit the model
    # history = model.fit(X_train_padded, y_train_padded, epochs=50, validation_data=(X_valid_padded, y_valid_padded), batch_size=64)

    # Save the model
    model.save('chatbot_v1.keras')

    #graph the training and validation loss
    import matplotlib.pyplot as plt
    history = history.history
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

