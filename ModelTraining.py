import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

## lemmatizer for convert word into base word
lemmatizer = WordNetLemmatizer()

## unique words
words = []

## unique tags
classes = []

## pattern, tag
documents = []

## ignore words
ignore_words = ['?', '!']

## load data from file
data_file = open("training_data/data.json").read()
intents = json.loads(data_file)


for intent in intents['intents']:
    # print(intent)
    for pattern in intent['patterns']:
        # print(pattern)
        
        w = nltk.word_tokenize(pattern)
        # print(w)
        words.extend(w)
        
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# print("documnets", documents)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

## save unique words 
pickle.dump(words, open('model_data/words.pkl', 'wb'))
## save all tags
pickle.dump(classes, open('model_data/classes.pkl', 'wb'))


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    
# print(training)

train_x = []
train_y = []

for i in training:
    train_x.append(i[0])
    train_y.append(i[1])
    
# print(len(train_x))
# print(len(train_y))

# shuffle our features and turn into np.array
# random.shuffle(training)
# training = np.array(training)
# create train and test lists. X - patterns, Y - intents
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])


print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

    
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.001, momentum=0.99, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



# # fitting and saving the mode
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=10, verbose=1)

model.save('model.keras')

# Plotting the training loss
plt.figure(figsize=(12, 6))
plt.plot(hist.history['loss'], label='Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss_graph.png')  # Save the loss graph as a PNG file
# plt.show()  # Display the graph

# Plotting the training accuracy
plt.figure(figsize=(12, 6))
plt.plot(hist.history['accuracy'], label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model_accuracy_graph.png')  # Save the accuracy graph as a PNG file
# plt.show()  # Display the graph


## trained a model with different different hyper parameter and the final model has this combination
# model_400_lr_01_momentum_99_nesterov_false_batch_size_10_accuracy_graph  accuracy: 0.9658 - loss: 0.0974
# model_400_lr_01_momentum_99_nesterov_True_batch_size_10_accuracy_graph : accuracy: 0.9775 - loss: 0.0967


if __name__ == "__main__":
    print("Done")