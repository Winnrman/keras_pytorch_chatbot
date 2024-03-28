from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from preprocess_data import preprocess_text
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from config import max_sequence_length

# Load the Parquet file into a DataFrame
df = pd.read_parquet('version1.parquet')

# Preprocess the data
prompts = df['prompt'].apply(preprocess_text).tolist()
responses = df['response'].apply(preprocess_text).tolist()

# Combine prompts and responses into a single list
combined_data = [' '.join([str(item) for item in pair]) for pair in zip(prompts, responses)]
# print(combined_data) #good

# Tokenize the combined data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_data)
word_index = tokenizer.word_index
# print(word_index)

# Create reverse_word_index
# reverse_word_index = {value: key for key, value in word_index.items()}
# print(reverse_word_index)

reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
# print(reverse_word_index)

# Load the model
model = load_model('chatbot_v1.keras')

# # Your testing logic here
# # User input loop
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         break

#     # Preprocess the user input
#     processed_input = preprocess_text(user_input)
#     # print(processed_input)

#     # Tokenize the input
#     tokenized_input = tokenizer.texts_to_sequences([processed_input])[0]
#     # print(tokenized_input) #this shouldn't be empty

#     # Pad the tokenized input
#     padded_input = pad_sequences([tokenized_input], maxlen=max_sequence_length, padding='pre')
#     # print(padded_input)

#     # Make prediction
#     prediction = model.predict(padded_input)
#     # print(prediction)

#     # Get the index of the highest probability
#     predicted_index = np.argmax(prediction[0])
#     # print(predicted_index)

#     # Print the response
#     print("AI:", reverse_word_index[predicted_index])


# Your testing logic here
# User input loop

# Beam search function
def beam_search_decoder(data, k):
    sequences = [[[], 0.0]]
    # Walk over each step in sequence
    for row in data:
        all_candidates = list()
        # Expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - np.log(row[j])]
                all_candidates.append(candidate)
        # Order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # Select k best
        sequences = ordered[:k]
    return sequences

# User input loop
# Your testing logic here
# User input loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Tokenize the input
    tokenized_input = tokenizer.texts_to_sequences([processed_input])[0]

    # Pad the tokenized input
    padded_input = pad_sequences([tokenized_input], maxlen=max_sequence_length, padding='pre')

    # Make prediction
    prediction = model.predict(padded_input)

    # Get the top 10 indexes of the highest probabilities
    predicted_indexes = np.argsort(prediction[0])[::-1][:5]

    # Create a list to store the words of the response
    response_words = []

    # Iterate over the top 20 indexes and append the corresponding words to the response list
    for index in predicted_indexes:
        response_words.append(reverse_word_index[index])

    # Join the words in the response list into a single sentence
    response_sentence = ' '.join(response_words)

    # Print the response
    print("AI:", response_sentence)

