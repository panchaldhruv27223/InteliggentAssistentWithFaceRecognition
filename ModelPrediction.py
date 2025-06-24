import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from keras.models import load_model
import os

print("Loading model...")
# print(os.getcwd())
# print("Current directory:", os.path.dirname(os.path.abspath(__file__)))
# print(os.listdir(os.getcwd()))

# import zipfile
# print(zipfile.is_zipfile("model.keras"))


model = load_model('model_data/model.keras')
intents = json.loads(open('training_data/data.json').read())
words = pickle.load(open('model_data/words.pkl', 'rb'))
classes = pickle.load(open('model_data/classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
                    
    return np.array(bag) 

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    # print(model.predict(np.array([p])))
    res = model.predict(np.array([p]))[0]
    # print("result",res)
    ERROR_THRESHOLD = 0.60
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print("result",results)
    
    results.sort(key=lambda x: x[1], reverse=True)
    # print("result : ",results)


    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        
    # print("return list : ",return_list)
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    if len(ints) == 0 :
        return "please say again." 
    # print("ints : ", ints)
    res = getResponse(ints, intents)
    return res


# print(chatbot_response("hello"))