import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import string

# def preprocess_text(text):
#     words = text.split()  # Split the text into words
#     # Remove punctuation from each word
#     words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
#     #words is a list like ['Who', 'are', 'you'] and we need [['who'],['are'],['you']]
#     words = [word.lower() for word in words]
#     return words

# def preprocess_text(text):
#     words = text.split()  # Split the text into words
#     # Remove punctuation from each word
#     words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
#     # Convert each word to lowercase and wrap in a list
#     words = [[word.lower()] for word in words]
#     # print(words)
#     return words #works

def preprocess_text(text):
    words = text.split()  # Split the text into words
    # Remove punctuation from each word
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    # Convert each word to lowercase
    words = [word.lower() for word in words]
    return ' '.join(words)  # Join the words back into a single string


def tokenize_data(data):
    # print(data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = max([len(seq) for seq in sequences])
    # print(vocab_size)
    # print(max_sequence_length)

    # print(sequences) #good
    # print(tokenizer.word_index)
    return sequences, vocab_size, max_sequence_length, tokenizer.word_index

def prepare_data(sequences, vocab_size, max_sequence_length):
    X = []
    y = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            X.append(sequence[:i])
            y.append(sequence[i])
    X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='pre')
    y_onehot = to_categorical(y, num_classes=vocab_size)
    return X_padded, y_onehot
