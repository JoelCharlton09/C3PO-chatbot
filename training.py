import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# nltk downloads
nltk.download("punkt_tab")
nltk.download("wordnet")

# Read the data from the intents.json file
data = json.loads(open("intents.json").read())

# Create lists for storing data
words = []              # Store each word individually from the intents file
classes = []            # Store the corresponding tag that each word belongs to
documents = []          # Store tokens alongside corresponding tags
X = []                  # Store all the patterns
y = []                  # Store each tag corresponding to the patterns in X
ignore_characters = ["!", ",", ".", "?"]                # Remove punctuation to avoid confusion for the model

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)        # Tokenize each pattern
        words.extend(tokens)                        # Put each token in the words list
        documents.append((tokens, intent["tag"]))   # Append tokens with corresponding tag
        X.append(pattern)                           # Append the full pattern to X
        y.append(intent["tag"])                     # Append the associated tag for each pattern to y
        
    # If a tag is not in classes, add it in
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
        
# Lemmatize the words
wnl = WordNetLemmatizer()
words = [wnl.lemmatize(word.lower()) for word in words if word not in ignore_characters]

# Sort words and classes making sure they're unqiue
words = sorted(set(words))
classes = sorted(set(classes))

# Create bag of words model
training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [wnl.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
        
    # Mark the index of the class the word belongs to
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Append the bag of words and associated class to training
    training.append([bag, output_row])
    
# Shuffle the data and convert to an array
random.shuffle(training)
training = np.array(training, dtype="object")

# Split data into features and labels
X_train = np.array(list(training[:,0]))
y_train = np.array(list(training[:,1]))

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation="softmax"))

# Stochastic gradient descent as the optimiser
sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)




### Preprocess the user input ###

# Convert text input to tokens
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [wnl.lemmatize(word) for word in tokens]
    return tokens

# Create bag of words for text input
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bag = [0]*len(vocab)
    for item in tokens:
        for idx, word in enumerate(vocab):
            if word == item:
                bag[idx] = 1
    return np.array(bag)

# Get the predicted tag based on probabilities
def pred_class(text, vocab, labels):
    bag = bag_of_words(text, vocab)
    result = model.predict(np.array([bag]))[0]  # Extract probabilities
    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)   # Sort by probability in decreasing order
    return_list = []
    for prob in y_pred:
        return_list.append(labels[prob[0]])
    return return_list

# Get a response from C3PO
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry! I cannot understand you."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for intent in list_of_intents:
            if intent["tag"] == tag:
                result = random.choice(intent["responses"])
                break
    return result

### Chatbot interaction ###
print("Press 0 if you don't want to chat to C-3PO.")
while True:
    message = input("")
    if message == "0":
        break
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)