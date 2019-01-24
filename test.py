
from keras.layers import Input, Dense
from keras.models import Sequential
from keras import losses
from keras.utils import plot_model
from keras.models import load_model
from math import log10
from router import Router
import numpy as np
import sys
sys.path.insert(0, './src')
from contentstore import DlcppContentStore

def get_entropy(req_hist):
    num_requests = sum(req_hist.values()) 
    request_entropy_array = []
    for content_type in req_hist.keys():
        content_request = req_hist[content_type]
        try:
            content_probability = content_request/num_requests
        except:
            content_probability = 0
        request_entropy_array.append(content_probability * log10(content_probability) if content_probability != 0 else 0)
    request_entropy = (-1) * sum(request_entropy_array)
    return request_entropy

def extract_features(req_hist):
    num_requests = sum(req_hist.values()) 
    content_sum = len(req_hist)
    request_entropy = get_entropy(req_hist)

    input_features = []

    for content_type in req_hist:
        content_request = req_hist[content_type]
        input_features.append([num_requests,content_request,content_sum,request_entropy])

    return np.array(input_features)

def get_true_labels(req_hist,pop_levels):
    #Calculate actual popularity **levels**(1 to 10) from previous time delta
    # if popularity is first = [1,0,0,0,0, ..,0,0]
    # if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
    req_hist_level = []
    num_requests_correct = sum(req_hist.values())
    for content_type in req_hist.keys():
        if req_hist[content_type] == 0:
            req_hist_level.append(pop_levels-1)
        else:
            req_hist_level.append(int((100*(1 - (req_hist[content_type] / num_requests_correct)) / pop_levels)) )#1st = 0, last = 9

    true_labels = np.zeros((len(req_hist_level),pop_levels), dtype=int) #2D array of [Content type] [pop_levels]
    true_labels[np.arange(len(req_hist_level)), req_hist_level] = 1 
    return true_labels



def train(model,contentstore,timestamp):
    pop_levels = 10
    input_features = extract_features(contentstore.prev_req_hist)
    true_labels = get_true_labels(contentstore.req_hist,pop_levels)
    print(input_features)
    print(true_labels)
    if not input_features.size == 0:
        model.fit(input_features,true_labels, epochs=100,verbose=2)
        score = model.evaluate(input_features,true_labels)
        print(score)
 

def predict(model, data):
    input_features = extract_features(data)
    prediction = model.predict(input_features)
    # print(prediction)
    print(np.argmax(prediction,axis = 1))

def baseline_model(n):
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(Dense(10, input_dim=8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def loaded_model(file):
    return load_model(file)

def report(model):
    print(model.summary())
    model.save('my_model.h5')
    plot_model(model, to_file='model.png')

            
        
            
        

        
            


