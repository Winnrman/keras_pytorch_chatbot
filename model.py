from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional
from tensorflow.keras.layers import Dropout


def create_model(vocab_size, max_sequence_length):

    model = Sequential()
    model.add(Input(shape=(max_sequence_length,)))
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(units=100))
    model.add(Dense(units=vocab_size, activation='softmax'))

    return model

# -------- More advanced model (Buggy with testing) --------

# def create_model(vocab_size, max_sequence_length):
#     model = Sequential()
#     model.add(Input(shape=(max_sequence_length,)))
#     model.add(Embedding(vocab_size, 100))
#     model.add(Bidirectional(LSTM(units=100)))  # Use Bidirectional wrapper
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(vocab_size, activation='softmax'))
#     return model







