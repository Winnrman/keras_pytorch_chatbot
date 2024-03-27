import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np
import string

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#preprocess_text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    return''.join(tokens)

#tokenize the data
def tokenize_data(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = max([len(seq) for seq in sequences])
    print("Before Tokenization - Input Shape:", len(data))
    print("Before Tokenization - Output Shape (sequences):", len(sequences))
    print("Before Tokenization - Vocab Size:", vocab_size)
    print("Before Tokenization - Max Sequence Length:", max_sequence_length)
    return sequences, vocab_size, max_sequence_length, tokenizer.word_index

#prepare the data
def prepare_data(sequences, vocab_size, max_sequence_length):
    X = []
    y = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            # Input sequence (history of tokens)
            X.append(sequence[:i])
            # Output sequence (next token to predict)
            y.append(sequence[i])
    # Pad sequences to have the same length
    X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='pre')
    # Convert y to one-hot encoded labels
    y_onehot = to_categorical(y, num_classes=vocab_size)
    print("After Tokenization - Input Shape (X_padded):", X_padded.shape)
    print("After Tokenization - Output Shape (y_onehot):", y_onehot.shape)
    return X_padded, y_onehot