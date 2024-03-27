from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

# def create_model(vocab_size, max_sequence_length):

#     model = Sequential()
#     model.add(Input(shape=(max_sequence_length,)))
#     model.add(Embedding(vocab_size, 100))
#     model.add(LSTM(units=100))
#     model.add(Dense(units=vocab_size, activation='softmax'))

#     return model

def create_model(vocab_size, max_sequence_length):

    model = Sequential()
    model.add(Input(shape=(max_sequence_length,)))
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(units=100))
    model.add(Dense(units=vocab_size, activation='softmax'))

    return model



